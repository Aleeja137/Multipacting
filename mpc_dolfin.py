"""
A library for multipacting calculations using FEniCS
"""
from dolfin import *
import numpy as np
import math
import sys
import scipy.constants as cte
from datetime import datetime
from vedo.dolfin import plot
from vedo import pointcloud
import hashlib
import matplotlib.pyplot as plt
import os

import concurrent.futures
#import ray

#ray.init()

class multipacting():
    """Class for a multipacting problem.
    
    """
       
    def __init__(self):
        self.data_file='' # EM field data, comsol format, at mesh nodes
        self.mesh_file='' # Mesh file, now h5 format. time will tell.
        self.mesh=None
        self.bmesh=None
        self.mesh_boundaries=None
        self.mesh_subdomains=None
        self.mesh_center=None
        self.tree=None
        self.btree=None

        self.data_ok=False
        self.mesh_ok=False
        self.campoEx=None
        self.campoEy=None
        self.campoEz=None

        self.N_ext=None
        self.lut_EX0=[]
        self.lut_E0=[]
        self.lut_sense=[]
        self.closest_entity_dictionary=dict()

        self.RF_frequency=0
        self.energy_0=1.0 # eV, energia cinética del electron que sale de la pared.
        self.N_cycles=20
        self.delta_t=1e-12
        
        self.e_m=-1.0*cte.e/cte.m_e
        self.tol_distance=1e-4
        
        self.angular_frequency=2*math.pi*self.RF_frequency
        self.electron_e_over_mc2=cte.e/(cte.m_e*cte.c*cte.c)
        self.c2=cte.c*cte.c
        self.electron_e_over_2mc=cte.e/(2.0*cte.m_e*cte.c)
        self.electron_e_over_2m=cte.e/(2.0*cte.m_e)
        
        self.solid_domains=[]
        self.domains_map=[]
        
        self.max_workers=8
        
        self.param={}
        
        self.N_runs_per_power=2000
        self.N_max_secondary_runs=15
        self.macro=10000
        
        self.verbose=True
        self.logfile_name=None
        self.logtime_name=None  #To keep execution times
        self.log=False
        self.show=False
        self.plot_title=''

        self.randSeed=-1 #Default: random seed.
        
    def set_parameters_dictionary (self, p):

        self.param=p
    
        self.log=p['log']
        self.N_runs_per_power=p['electrons_seed']
        self.N_max_secondary_runs=p['N_max_secondary_runs']
        self.verbose=p['verbose']
        self.randSeed=p['random_seed']
        
        if 'range' in p['RF_power']:
            ps=p['RF_power']
            ps=ps[ps.find("(")+1:ps.find(")")]
            x=ps.split(',')
            self.RF_power=np.arange(float(x[0]),float(x[1]),float(x[2]))
            self.param['RF_power']=self.RF_power
        else:
            x=float(p['RF_power'])
            self.param['RF_power']=np.array([x])
            
        self.RF_frequency=p['RF_frequency']
        self.angular_frequency=2*math.pi*self.RF_frequency
        self.delta_t=p["delta_t"]
        self.N_cycles=p["N_cycles"]
        
        if len(p['plot_title'])>0:
            self.plot_title=p['plot_title']
            
        if 'comsol_solid_domains' in p.keys():
            self.set_solid_domains_COMSOL (p['comsol_solid_domains'])

    def get_N_surface_elements (self):
        return self.N_ext

    def read_mesh_file (self, mesh_file=None):
        if mesh_file:
            self.mesh_file=mesh_file

        try:
            # read mesh
            #
            parameters["reorder_dofs_serial"]=False # Extremely important
            #
            self.mesh = Mesh()
            hdf = HDF5File(self.mesh.mpi_comm(), self.mesh_file, 'r')
            hdf.read(self.mesh, "/mesh", False)
            self.mesh_subdomains=MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
            hdf.read(self.mesh_subdomains, "/subdomains")
            self.mesh_boundaries=MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
            hdf.read(self.mesh_boundaries, "/boundaries")
            

            # mesh analysis
            self.N_cells=self.mesh.num_cells() # tetras
            
            self.bmesh=BoundaryMesh(self.mesh,'exterior')
            self.N_ext=self.bmesh.num_cells()
            self.mesh_center=self.mesh_center_point (self.bmesh)
            
            self.tree=BoundingBoxTree()
            self.tree.build(self.mesh,3)
            
            self.btree=BoundingBoxTree()
            self.btree.build(self.bmesh,2)
            
            self.N_elems=self.N_ext # surface elements
            
            self.domains_map=self.mesh_subdomains.array()

            self.lut_sense=[False]*self.N_ext

            self.mesh_ok=True
            
        except:
            print (f'Error with mesh file {self.mesh_file}.')
            self.mesh_ok=False
            sys.exit(1)
            
        # print("mesh_file: {}, type: {}\n".format(self.mesh_file,type(self.mesh_file)))
        # print("N_cells: {}, type: {}\n".format(self.N_cells,type(self.N_cells)))
        # print("N_ext: {}, type: {}\n".format(self.N_ext,type(self.N_ext)))
        # print("N_elems: {}, type: {}\n".format(self.N_elems,type(self.N_elems)))
        # print("mesh_ok: {}, type: {}\n".format(self.mesh_ok,type(self.mesh_ok)))
        
        # print("mesh_center: {}, type: {}, n_elem: {}, each element: \n".format(self.mesh_center,type(self.mesh_center),len(self.mesh_center)))
        # for element in self.mesh_center:
        #     print("{}, type {}; ".format(element,type(element)))
        
        # print("domains_map: {}, type: {}, n_elem: {}, each element: \n".format(self.domains_map,type(self.domains_map),len(self.domains_map)))
        # # for element in self.domains_map:
        # #     print("{}, type {}; ".format(element,type(element)))
        
        # print("lut_sense, type: {}, n_elem: {}, each element: \n".format(type(self.lut_sense),len(self.lut_sense)))
        # # for element in self.lut_sense:
        # #     print("{}, type {}; ".format(element,type(element)))
            
        # print("Type of mesh: {}, geometry: {},n_vértices: {},n_celdas: {},n_facets: {}\n".format(type(self.mesh),self.mesh.geometry().dim(),self.mesh.num_vertices(),self.mesh.num_cells(),self.mesh.num_facets()))
        
        # print("Mesh_subdomains type: {},n_values: {}, each element: \n".format(type(self.mesh_subdomains),len(self.mesh_subdomains.array())))
        # # for element in self.mesh_subdomains.array():
        # #     print("{}, type {}; ".format(element,type(element)))
        
        # print("mesh_boundaries type: {},n_values: {}, each element: \n".format(type(self.mesh_boundaries),len(self.mesh_boundaries.array())))
        # # for element in self.mesh_boundaries.array():
        # #     print("{}, type {}; ".format(element,type(element)))
            
        # print("bmesh type: {},n_vértices: {}, n_celdas: {}\n".format(type(self.bmesh),self.bmesh.num_vertices(),self.bmesh.num_cells()))
        
    def mesh_center_point(self, mesh=None):
        if not mesh:
            mesh=self.mesh

        mc=mesh.coordinates()
        
        minx=0
        maxx=0
        miny=0
        maxy=0
        minz=0
        maxz=0
        for nc in mc:
            x=nc[0]
            y=nc[1]
            z=nc[2]
            minx=min(minx,x)
            miny=min(miny,y)
            minz=min(minz,z)
            maxx=max(maxx,x)
            maxy=max(maxy,y)
            maxz=max(maxz,z)

        median_x=(maxx+minx)/2.0
        median_y=(maxy+miny)/2.0
        median_z=(maxz+minz)/2.0
    
        Xc=median_x
        Yc=median_y
        Zc=median_z

        return [Xc,Yc,Zc]

    def set_solid_domains_COMSOL (self, ld):
        self.solid_domains=[comsol_domain+1 for comsol_domain in ld]

    def domain_histogram (self, show=False):
        md=self.mesh_subdomains.array()
        dh={}
        for n in md:
            if n in dh.keys():
                dh[n]=dh[n]+1
            else:
                dh[n]=1
                
        if show:
            print(dh)
            
        return dh

    def read_field_data (self,build_lookup_table=True):
        data_file=self.data_file

        try:
            # read E B field and process
            datosEB=np.genfromtxt (data_file, skip_header=9, dtype=str)
            mapping=np.vectorize(lambda t:complex(t.replace('i','j')))
            datosEB = mapping(datosEB)
            
            # print("\n---datosEB---\n")
            # for dato in datosEB:
            #     print(dato)

            # Real part
            datosEBn=np.real(datosEB)
            # print("\n---datosEBn---\n")
            # for dato in datosEBn:
            #     print(dato)

            X=datosEBn[:,0]
            Y=datosEBn[:,1]
            Z=datosEBn[:,2]
            EX=datosEBn[:,3]
            EY=datosEBn[:,4]
            EZ=datosEBn[:,5]
            
            # print("\n---Datos EX---\n")
            # np.set_printoptions(threshold=np.inf)
            # np.set_printoptions(floatmode='fixed', precision=6)
            # print(EX)
            # print("\n---Fin datos EX---\n")
            # print("Size of EZ: {}\n".format(len(EZ)))
            
            # sys.exit(0)
            
            # print("\n---Datos Y---\n")
            # print(Y)
            # print("\n---Datos Z---\n")
            # print(Z)
            # print("\n---Datos EX---\n")
            # print(EX)
            # print("\n---Datos EY---\n")
            # print(EY)
            # print("\n---Datos EZ---\n")
            # print(EZ)
            

            # FEniCS black magic
            # I create a finite element function space on the mesh I have read.
            # field data will be interpolated (projected) into that vector space

            V=FunctionSpace (self.mesh,"CG",1)
            # Defino las funciones para el campo dentro del espacio de funciones V
            Fez=Function(V)
            Fez.vector().set_local(EZ) # <---- This is the mother of the lamb
            Fiez=interpolate(Fez,V)
            Fiez2=project(Fez,V)
            Fex=Function(V)
            Fex.vector().set_local(EX)
            Fey=Function(V)
            Fey.vector().set_local(EY)
            
            # np.savetxt('EX_numpy.txt',EX)
            # np.savetxt('EY_numpy.txt',EY)
            # np.savetxt('EZ_numpy.txt',EZ)
            
            # funciones para interpolar
            campoEx=project(Fex,V)
            campoEx.set_allow_extrapolation(True)
            campoEy=project(Fey,V)
            campoEy.set_allow_extrapolation(True)
            campoEz=project(Fez,V)
            campoEz.set_allow_extrapolation(True)

            # campoEx_cpp = Function(V)
            # with HDF5File(self.mesh.mpi_comm(), "/home/alejandro/Desktop/Universidad/DIPC/4_mpc_python_tests/bin/campoEx.h5",'r') as f:
            #     f.read(campoEx_cpp,'/campoEx')
            # err = np.sqrt(assemble((campoEx_cpp - campoEx)**2*dx))
            # print('Error EX:', err)
            
            # campoEy_cpp = Function(V)
            # with HDF5File(self.mesh.mpi_comm(), "/home/alejandro/Desktop/Universidad/DIPC/4_mpc_python_tests/bin/campoEy.h5",'r') as f:
            #     f.read(campoEy_cpp,'/campoEy')
            # err = np.sqrt(assemble((campoEy_cpp - campoEy)**2*dx))
            # print('Error EY:', err)
            
            # campoEz_cpp = Function(V)
            # with HDF5File(self.mesh.mpi_comm(), "/home/alejandro/Desktop/Universidad/DIPC/4_mpc_python_tests/bin/campoEz.h5",'r') as f:
            #     f.read(campoEz_cpp,'/campoEz')
            # err = np.sqrt(assemble((campoEz_cpp - campoEz)**2*dx))
            # print('Error EZ:', err)
            # sys.exit(0)
            
            with HDF5File(self.mesh.mpi_comm(), "/home/alejandro/Desktop/Universidad/DIPC/4_mpc_python_tests/bin/campoEx.h5",'r') as f:
                f.read(campoEx,'/campoEx')
            
            with HDF5File(self.mesh.mpi_comm(), "/home/alejandro/Desktop/Universidad/DIPC/4_mpc_python_tests/bin/campoEy.h5",'r') as f:
                f.read(campoEy,'/campoEy')
            
            with HDF5File(self.mesh.mpi_comm(), "/home/alejandro/Desktop/Universidad/DIPC/4_mpc_python_tests/bin/campoEz.h5",'r') as f:
                f.read(campoEz,'/campoEz')
                
            print("Usando el mismo campo que la versión C++")
            
            self.campoEx=campoEx
            self.campoEy=campoEy
            self.campoEz=campoEz
            
            # valuesEx = self.campoEx.vector().get_local()
            # valuesEy = self.campoEy.vector().get_local()
            # valuesEz = self.campoEz.vector().get_local()
            # print("Valores del campoEx:")
            # print(valuesEx)
            # print("Valores del campoEy:")
            # print(valuesEy)
            # print("Valores del campoEz:")
            # print(valuesEz)
            
            
            if not build_lookup_table:
                return True
            
            print(self.N_ext)
            # Some field data pre-processing to accelerate calculations
            for i in range(self.N_ext): 
                facet_i=cpp.mesh.Face(self.bmesh,i)
                mp=facet_i.midpoint()
                X0=[mp[0],mp[1],mp[2]]
                area=facet_i.area()
                # print("Para i:{},mp es: {}, con area: {}\n".format(i,X0,area))

                # result = self.campoEx(X0)
                # print("Result (EX) is type: {}; and value: {}\n".format(type(result),result))
                EX0=np.array([self.campoEx(X0), self.campoEy(X0), self.campoEz(X0)])
                E0=np.sqrt(np.sum(np.square(EX0)))
                self.lut_EX0.append (EX0)
                self.lut_E0.append (E0)
                # print("i: {} E0: {}".format(i,E0))
                # print("i: {} X0: {}".format(i,X0)) # Correcto
                # print("i: {} EX0: {}".format(i,EX0))

            # print(self.lut_EX0)
            # print(self.lut_E0)            
            return True

        except:
            print ("Read_field_data: Error reading data file.")
            sys.exit(1)
            return False

    def read_from_data_files (self, mesh_file, data_file):
        
        self.data_file=data_file
        self.mesh_file=mesh_file

        self.read_mesh_file ()
        self.read_field_data ()
        
        if self.verbose:
            print ('Data file: ', data_file)
            print ('Mesh file; ', mesh_file) 
            print (f'Data read. Num surface elems: {self.N_elems}, Num volume elems; {self.N_cells}\n')
           
    def read_input_files (self):
        self.read_from_data_files (self.param['mesh_file'],self.param['data_file'])
        
    def plot_surface_mesh (self):
        vedo_plotter=None
        if self.bmesh:
            vedo_plotter=plot (self.bmesh, color='gray', wireframe=True, interactive=False)

        return vedo_plotter

    def closest_entity (self, X):
        strx=f'{X[0]:.4f}_{X[1]:.4f}_{X[2]:.4f}'
        #strh=int(hashlib.sha256(strx.encode('utf-8')).hexdigest(),16)
        #print (strx, strh)
        if strx in self.closest_entity_dictionary:
            D=self.closest_entity_dictionary[strx]
        else:
            D=self.tree.compute_closest_entity(Point(X))
            self.closest_entity_dictionary[strx]=D

        return D

    def point_inside_mesh (self, X):
    
        ent=self.tree.compute_first_entity_collision(Point(X))
        if ent >= self.N_cells:
            return False
                           
        if self.solid_domains:    
            domain_index=self.domains_map[ent]
            if domain_index in self.solid_domains: # Point inside a solid domain
                return False

        return True
            
    def get_initial_conditions_face (self, face_i):
        # face_i = 2099
        nodes=self.bmesh.cells()[face_i]
        # print(nodes)
        # sys.exit(0)
        facet_i=cpp.mesh.Face(self.bmesh,face_i)

        X=[self.bmesh.coordinates()[nodes[i]] for i in range(3)]
        # print(X)
        # sys.exit(1)
        mp=facet_i.midpoint()
        # print("face_i: {} mp is: {},{},{}".format(face_i,mp.x(),mp.y(),mp.z()))
        Nv=face_normal (X)
        # print("Nv is: {}".format(Nv))

        # Element sense (normal pointing inward or outward)
        sense_factor=self.lut_sense[face_i]
        if not sense_factor:
            # Tengo que calcularlo
            Xc=self.mesh_center
            dcm=np.sqrt((mp[0]-Xc[0])**2+(mp[1]-Xc[1])**2+(mp[2]-Xc[2])**2)

            # Avanzo 0.1*distancia en la dirección de la normal:
            Xp=mp[0]+(dcm*0.1*Nv[0])
            Yp=mp[1]+(dcm*0.1*Nv[1])
            Zp=mp[2]+(dcm*0.1*Nv[2])
            #        new_dcm=np.sqrt((Xp-Xc[0])**2+(Yp-Xc[1])**2+(Zp-Xc[2])**2)
            dcm_e,dcm_d=self.closest_entity([Xp,Yp,Zp])

            sense_factor=1
            if dcm_d>0.0:
                # punto fuera del mallado
                sense_factor=-1
                # the electron is emitted to the interior of cavity from the surface
                # mesh. BoundaryMEsh is not oriented, so we need to compute the inward
                # direction by this trick

            # update look-up table
            self.lut_sense[face_i]=sense_factor 
        # print("Sense_factor: ",sense_factor)
        # tracking        
        # starting conditionsn

        X0=[mp[0],mp[1],mp[2]]
        U0=np.array([Nv[0],Nv[1],Nv[2]])*sense_factor
        EX0=self.lut_EX0[face_i]
        
        return X0,U0,EX0
        
    def track_1_e (self, electron_energy=None, power=1.0, phase=0.0, face_i=None, keep=False, show=False, starting_point=None):
        ''' Runs the tracking of 1 electron in the problem geometry.
        power: RF power [W] in the device

        phase: phase [rad] when eletron is emmited (field will be E=E0 cos (wt+phase)

        face_i: the surface facet element where the electron is emmited. If not
        especified it is chosen ramdomly.

        keep: [False] Wheather or not to keep the full trayectory. If
        show==True, this is automatically also True

        show: [False] Shows the mesh and the electron trayectory.

        Return values: [collision, energy_collision]
            collision: face index where electron ended (or None)
            energy_collision: energy [eV] of the electron when collision happens

        '''
        # keep = True
        face_i = 1102
        # show = True

        if self.log:
            logfile=open (self.logfile_name, 'a')
            logfile.write (f'Call track_1_e starting from face={face_i}, phase={phase}, energy={electron_energy}\n')
            logfile.close ()

        
        magnetic_field=self.param["magnetic_field_on"]
        # field read from file taken from look-up table
        campoEx=self.campoEx
        campoEy=self.campoEy
        campoEz=self.campoEz
        field_factor=math.sqrt(power)
        
        if magnetic_field:
            campoBx=self.campoBx
            campoBy=self.campoBy
            campoBz=self.campoBz

        trayectoria=None
        collision=None
        energy_collision=None
        phase_collision=None
        if not electron_energy:
            electron_energy=self.energy_0
            
        max_gamma=0

        if show:
            keep=True
            
        V0=None
        U0=None
        
        if not starting_point:
            if not isinstance(face_i,int):
                face_i=np.random.randint (0,self.N_ext)
        
            # face_i = 1814  # AQUIIIIIIIIIIIIII
            print("Starting with face_i: ",face_i)
            X0,U0,EX0=self.get_initial_conditions_face (face_i)   
        
            # print("X0 es : {}".format(X0))
            # print("U0 es : {}".format(U0))
            # print("EX0 es: {}".format(EX0))
            # sys.exit(1)
        else:
            X0=starting_point[0]
            EX0=np.array([campoEx(X0), campoEy(X0), campoEz(X0)])
            if len(starting_point)>1:
                V0=np.array(starting_point[1])
                beta=np.linalg.norm(V0)/cte.c
                gamma=1.0/np.sqrt(1-beta*beta)

        # Initial velocity: taken from starting_point[1]. If not available, computed from electron_energy           
        if not isinstance(V0,np.ndarray):
            gamma=1.0+electron_energy*self.electron_e_over_mc2
            beta=math.sqrt(gamma*gamma-1)/gamma
            # print(cte.e)
            # print(cte.m_e)
            # print(cte.c)
            # print(self.electron_e_over_mc2)
            # print(gamma)
            # print(beta)
            v0=beta*cte.c # en m/s
            #V0=np.array([Nv[0],Nv[1],Nv[2]])
            
            if not isinstance (U0,np.ndarray): # random direction
                U0=np.random.rand(3)*2-1 
                U0=U0/np.linalg.norm(U0)
            
            V0=U0*v0
            # print("U0 type: {} and values: {}".format(type(U0),U0))
            # print("v0 type: {} and values: {}".format(type(v0),v0))
            # print("V0 type: {} and values: {}".format(type(V0),V0))
            # sys.exit(0)


        t=0
        
        w=self.angular_frequency

        alpha=w*t+phase
        EX=EX0*field_factor*np.cos(alpha)

        #
        # tracking
        #
        
        ended=False
        VX=V0
        X=X0
        delta_t=self.delta_t
        t_max=self.N_cycles/self.RF_frequency
        
        if keep:
            trayectoria=[]
            energia_electron=[]
            
        #X=[0.010909800696347562, 0.00835580990654104, 0.003188495412826358] 
        #VX=np.array([ -23778.25308883,  264720.64515719, -537919.67309962])
        #print (X,VX)
        #print (phase)

        # Boris integrator, with B if required
        P0=(gamma/cte.c)*V0
                   
        # print("VX is: ",VX)
        # print("V0 is: ",V0)
        # print("EX is: ",EX)
        # print("EX0 is: ",EX0)
        # print("X is: ",X)
        # print("X0 is: ",X0)
        # print("U0 is: ",U0)
        # print("delta_t is: ",delta_t)
        # print("t_max is: ",t_max) 
        # print("P0 is: ",P0)
        # print("alpha is: ",alpha)
        # print("gamma is: ",gamma)
        # print("w is: ",w)
        # print("power is: ",power)
        # print("field_factor is: ",field_factor)
        # print("t is: ",t)
        
        while not ended:
            # E field only
            EX0=np.array([campoEx(X), campoEy(X), campoEz(X)])
            alpha=w*t+phase
            EX=EX0*field_factor*np.cos(alpha)
            
            if not self.param["magnetic_field_on"]:
                P_minus=P0-self.electron_e_over_2mc*EX*delta_t
                gamma_minus=math.sqrt(1.0+np.dot(P_minus,P_minus))

                # No B
                P_plus=P_minus   
                
                P=P_plus-self.electron_e_over_2mc*EX*delta_t
                gamma=math.sqrt(1+np.dot(P,P))

                #VX=P/(gamma*cte.m_e)      
                VX=cte.c*P/gamma      
                P0=P        
                       
            else:
                BX=np.array([campoBx(X), campoBy(X), campoBz(X)])
                #BX=np.array([0.0964,0,0])
                
                U0=VX*gamma
                epsilon=(-1)*self.electron_e_over_2m*EX
                T = (-1)*self.electron_e_over_2m*delta_t*BX/gamma
                
                U_minus=U0+epsilon*delta_t
                U_prime=U_minus+np.cross(U_minus,T)
                s=2.0/(1.0+np.linalg.norm(T)**2)
                U_plus=U_minus+s*np.cross(U_prime,T)
                
                UX=U_plus+epsilon*delta_t
                
                VX = UX/gamma
                
                beta=np.linalg.norm(VX)/cte.c
                gamma=1.0/np.sqrt(1-beta*beta)


            max_gamma=max(gamma,max_gamma)   
            
            X=X+VX*delta_t
            V0=VX
            
            if not self.point_inside_mesh(X):
                # out of the mesh, this is a collision with a boundary
                # vn=np.sqrt(VX[0]*VX[0]+VX[1]*VX[1]+VX[2]*VX[2])
                # beta=vn/cte.c
                # gamma=1/(1-(beta*beta))


                energia_eV=(gamma-1)/self.electron_e_over_mc2
                if gamma<=1.0:
                    print ('gamma=', gamma, 'eV=', energia_eV)
                    print (VX[0],VX[1],VX[2])
                    sys.exit(1)

                #cell_i,d=self.closest_entity(X)
                
                collision_face_i,ds=self.btree.compute_closest_entity(Point(X))
                while collision_face_i>=self.N_ext:
                    print (f'Fallo de btree, con X={X} y collision={collision_face_i},{ds}')
                    xyz=np.random.randint(3)
                    s=1
                    sr=np.random.rand()*0.002-0.001
                    Y=[0,0,0]
                    Y[xyz]=sr
                    X=X+np.array(Y)
                    collision_face_i,ds=self.btree.compute_closest_entity(Point(X))
                    
                    
                collision=collision_face_i
                energy_collision=energia_eV
                phase_collision=alpha

                ended=True

            if keep:
                energia_eV=(gamma-1)/self.electron_e_over_mc2
                trayectoria.append(X)
                energia_electron.append(energia_eV)
                #print (X,VX,EX0)

            t=t+delta_t
            if t>t_max:
                ended=True
        
        if show:
            #Z=np.arange(-0.08,0.08,0.0001)
            #trayectoria=[[0,0,z] for z in Z]
            vp=self.plot_surface_mesh()
            vp+=pointcloud.Points(trayectoria, r=3, c='red')
            vp.show(interactive=True)
            # plt.plot (energia_electron)
            # plt.ylabel ('electron energy (eV)')
            # plt.show()
            
        if keep:
            ftraj=open ('generated_files/ultima_trayectoria.txt', 'w')
            for x,y,z in trayectoria:
                ftraj.write ('%f\t%f\t%f\n' % (x,y,z))
                
            np.savetxt ('generated_files/ultima_energia.txt', energia_electron)

            ahora=datetime.now()            
            name_exp=f'generated_files/{ahora.isoformat()}.txt'
            
            np.savetxt (name_exp, np.c_[np.array(trayectoria), energia_electron])
            

        if self.log:
            logfile=open (self.logfile_name, 'a')
            logfile.write (f'Completed, collision={collision}\n')
            logfile.close ()
            
        if not collision:
            energia_eV=(max_gamma-1)/self.electron_e_over_mc2
            energy_collision=energia_eV

        #print (f'Completed run {collision}, {energy_collision}')
        return collision, energy_collision,phase_collision,trayectoria

    def total_secondary_electrons (self, energy_eV):

        sey=self.secondary_electron_yield (energy_eV)
        n=self.probability_of_emmision (sey)
        energies=[]
        bote=energy_eV
        for x in range (n):
            y=np.random.rand()
            e0=y*bote
            energies.append(e0)
            bote=bote-e0

        return n, energies, sey

    def secondary_electron_yield (self, energy_eV):
        ''' Simple SEY formulae, intended to be overruled by the user.
        '''
        ev=energy_eV
        try:
            #S=0.005*ev*math.exp(-0.001*ev)*1.2
            S=0.003*np.power(ev,1.3)*np.exp(-0.003*ev)+0.2+0.5*np.exp(-0.01*ev)

        except OverflowError:
            print ('Overflow in exponential math.exp. Energy=',ev)
            S=0
        return S
         
    def efn_emmision (self, power=1.0):
        
        def electrons_from_facet (x):
            facet_i=cpp.mesh.Face(self.bmesh,x)
            area=facet_i.area()
            
            field_factor=math.sqrt(power)

            E0=self.lut_E0[x]
            E0*=field_factor
            
            jfn=fowler_nordhaim_current_density (E0,beta=200) # A/m2
            ifn=jfn*area    # A
            
            ne_fn=ifn/(cte.e*self.RF_frequency)    # num_electrones en 1 ciclo RF
            return int(ne_fn)
            
        eem=[electrons_from_facet(x) for x in range(self.N_ext)]
        return eem

    def remove_by_coordinate_value (self,lista_f, coordinate, value):
        new_lista=[]
        for s in lista_f:
            facet_i=cpp.mesh.Face(self.bmesh,s)
            mp=facet_i.midpoint()
            if not (abs(mp[coordinate]-value))<1e-6:
                new_lista.append(s)    

        return new_lista
        
    def remove_by_boolean_condition (self,lista_f, operation, coordinate, value, tol=1e-6):
        new_lista=[]
        for s in lista_f:
            facet_i=cpp.mesh.Face(self.bmesh,s)
            mp=facet_i.midpoint()
            
            # check condition
            condition=False
            x=mp[coordinate]
            if operation==0: # equal
                condition=abs(x-value)<tol
            if operation==-1: # less than
                condition=x<value
            if operation==+1: # more than
                condition=x>value
                
            if not condition:
                new_lista.append(s)
                
            
        return new_lista

    def probability_of_emmision (self, sey):

        if self.randSeed != -1:
            np.random.seed(self.randSeed+1) #To generate the same electrons regardless of the order in which they are simulated
        x=np.random.rand()
        y=0
        n=0
        n_fact=1
        while y<x:
            p=math.pow(sey,n)*math.exp(-sey)
            p/=n_fact
            y+=p
            n=n+1
            n_fact=n_fact*n

        return n-1

    def run_1_electron (self, power, face, rf_phase, energy, keep=False):
       
        start_1e = datetime.now()

        y=self.track_1_e (electron_energy=energy, power=power, phase=rf_phase, face_i=face, keep=keep)

        end_1e = datetime.now()

        #Guardar tiempo 1 electron       
        segundos_1e = ((end_1e.minute-start_1e.minute)*60) + (end_1e.second-start_1e.second) + ((end_1e.microsecond - start_1e.microsecond)/1000000) + ((end_1e.hour-start_1e.hour)*360)
        logtime=open (self.logtime_name, 'a')
        if (segundos_1e >= 1):
            logtime.write (f'time 1e: {format(segundos_1e,".3f")} sec \n')
        else:
            milisegundos_1e = segundos_1e*1000
            logtime.write (f'time 1e: {format(milisegundos_1e,".3f")} ms \n')
        logtime.close ()

        #Mostrar tiempo 1 electron
        #print ("time 1e:", format(segundos_1e,".3f"), "sec")

        return y

    def run_n_electrons_parallel (self, power, pool_runs, pool_phase,
            pool_energies, logfile_name=None, logtime_name=None):

        logfile_name=self.logfile_name
        logtime_name=self.logtime_name
        log=(logfile_name!=None)

        ended=False

        #Mostrar y guardar estado inicial
        print(f'Power={power} W, initial #electrons: {len(pool_phase)}\n')
        logtime=open (self.logtime_name, 'a')
        logtime.write (f'Power={power} W, initial #electrons: {len(pool_phase)}\n')
        logtime.close ()

        if log:
            logfile=open (logfile_name, 'a')
            logfile.write (f'Power={power} W, initial #electrons: {len(pool_phase)}\n')
            logfile.close()
    
        if self.show:
            vp_color='red'
            vp=self.plot_surface_mesh()
    
        lut_results={}
    
        n=0
        number_electrons=0
        while not ended:

            start_run = datetime.now()

            new_pool_runs=[]
            new_pool_phases=[]
            new_pool_energies=[]

            if len(pool_runs)==0:
                ended=True
                continue

            mw=self.param['parallel']
            if len(pool_runs)<mw:
                mw=len(pool_runs)

            if mw==1: # Serial
                for erun,efase,energy_0 in zip(pool_runs,pool_phase,pool_energies):

                    lut_index=f'{erun}_{efase:.4f}_{energy_0:.4f}'
                    
                    if lut_index in lut_results.keys():
                        face,energy,phase,trayectoria=lut_results[lut_index]
                        
                    else:
                        face,energy,phase,trayectoria=self.run_1_electron (power,erun,efase,energy_0,keep=self.show)
                        lut_results[lut_index]=(face,energy,phase,trayectoria)
                      
                    if face:
                        n_e,energies,sey=self.total_secondary_electrons (energy)
                    
                        if log:
                            logfile=open (logfile_name, 'a')
                            logfile.write (f'\tCompleted run in face {face}, energy={energy} eV, phase={phase} rad\n')
                            logfile.write (f'\tThis produces {sey} sey and {n_e} new electrons\n')
                            logfile.write ('\n')
                            logfile.close()
                    
                        for _ in range (n_e):
                            new_pool_runs.append(face)
                            new_pool_phases.append(phase)

                        new_pool_energies.extend(energies)
                        
                        
            else: # mw>1 Parallel
            
                RAY_LIB=False
                MP_LIB=not RAY_LIB
                
                # using ray library
                if RAY_LIB:
                    trackers=[self.run_1_electron.remote() for i in range(mw)]
                    
                    break
            
                with concurrent.futures.ThreadPoolExecutor(max_workers=mw) as executor:
                    results = [executor.submit(self.run_1_electron,
                        power,erun,efase,energy_0,keep=self.show) for erun,efase,energy_0 in zip(pool_runs,pool_phase,pool_energies)]
                
                for f in concurrent.futures.as_completed(results):
                                
                    face=f.result()[0]
                    energy=f.result()[1]
                    phase=f.result()[2]
                    trayectoria=f.result()[3]
                                
                    if face:
                        n_e,energies,sey=self.total_secondary_electrons (energy)
                
                        if log:
                            logfile=open (logfile_name, 'a')
                            logfile.write (f'\tCompleted run in face {face}, energy={energy} eV, phase={phase} rad\n')
                            logfile.write (f'\tThis produces {sey} sey and {n_e} new electrons\n')
                            logfile.write ('\n')
                            logfile.close()

                        for _ in range (n_e):
                            new_pool_runs.append(face)
                            new_pool_phases.append(phase)

                        new_pool_energies.extend(energies)

            pool_runs=new_pool_runs
            pool_phase=new_pool_phases
            pool_energies=new_pool_energies

            #Guardar tiempo de una "run"
            end_run = datetime.now()
            minutos_run = end_run.minute - start_run.minute + ((end_run.second - start_run.second)/60) + ((end_run.microsecond-start_run.microsecond)/60000000) + ((end_run.hour-start_run.hour)*60)
            logtime=open (self.logtime_name, 'a')
            if (minutos_run >= 1):
                logtime.write (f'Completed run {n}, time: {format(minutos_run,".3f")} min, electrons alive: {len(pool_runs)} \n')
            else:
                segundos_run = minutos_run*60
                logtime.write (f'Completed run {n}, time: {format(segundos_run,".3f")} sec, electrons alive: {len(pool_runs)} \n')
            logtime.close ()

            #Mostrar tiempo de una "run"
            print ("Time of run:", format(minutos_run,".3f"), "min")

            print (f'Power={power} W, run#: {n}, electrons alive:{len(pool_runs)}')
            
            if log:
                logfile=open (logfile_name, 'a')
                logfile.write (f'Completed secondary run #: {n}, power={power} W, electrons alive:{len(pool_runs)}\n')
                logfile.write ('\n')
                logfile.close()
                
            if self.show:
                vp+=pointcloud.Points(trayectoria, r=3, c=vp_color)
                if vp_color=='red':
                    vp_color='green'

                   
            number_electrons=number_electrons+len(pool_runs)

            n=n+1
            if n>self.N_max_secondary_runs:
                ended=True
                print ("Max number of secondary runs achieved at P=%f W" % power) 

                if log:
                    logfile=open (logfile_name, 'a')
                    logfile.write ("Max number of secondary runs achieved at P=%f W\n" % power) 
                    logfile.close()

            if len(pool_runs)==0:
                ended=True

        if self.show:                   #Comentar para evitar tener que cerrar la imagen para que termine el programa
            vp.show(interactive=True)

        return number_electrons, len(pool_runs)
        
    def run (self):
        
        start_total = datetime.now()
        # self.randSeed = -1 # AQUIIIIIIIIIIIIIIII
        if self.randSeed==-1:
            np.random.seed()
        else:
            np.random.seed(self.randSeed)

        log=self.param["log"]
        logfile_name=self.logfile_name
        logtime_name=self.logtime_name
        now=datetime.now()
        now_str=now.strftime ("%Y%m%d_%H%M%S")
        if log and self.param['simulation_type']!=2:
            self.logfile_name='generated_files/log_mpc_py_%s.txt' % now_str
            logfile=open (self.logfile_name, 'w')
            logfile.close()

        self.logtime_name='generated_files/exec_time_%s.txt' % now_str
        logtime=open (self.logtime_name, 'w')
        logtime.close()
    
        self.param['simulation_type'] = 2 # AQUIIIIIIIIIIIIIIII
        if self.param['simulation_type']==2:
            rf_power=(self.param['RF_power'])[0]

            test_face=[]
            while len(test_face)<1:
                test_face.append (np.random.randint (0,self.N_ext))
                ## Remove by coordinate, boolean
                for ct in self.param['boundaries_excluded_boolean']:
                    tol=1e-6
                    operation=ct[0]
                    coordinate=ct[1]
                    value=ct[2]
                    if len(ct)>3:
                        tol=ct[3]
                    test_face=self.remove_by_boolean_condition (test_face, operation, coordinate, value,tol)

            trayectoria=self.track_1_e (power=rf_power, show=True, face_i=test_face[0])
            return
            
        if self.param['simulation_type']==3:
            rf_power=(self.param['RF_power'])
            self.N_runs_per_power=1
            #self.show=True            #Comentar para ejecutarlo en atlas
        if self.param['simulation_type']==1:
            self.N_runs_per_power=self.param["electrons_seed"]
            rf_power=self.param['RF_power']
        
        print("\n---Imprimiendo power---\n")
        print(rf_power)
        
        total_electrons=[]
        final_electrons=[]
        power_partial=[]
        for power in rf_power:
                
            pool_runs=[]
            #np.random.seed()
            new_pool_runs=[np.random.randint (0,self.N_elems) for _ in range(self.N_runs_per_power)]
            print("power: {}, new_pool_runs: {}\n".format(power,new_pool_runs))
            print(rf_power)
                       
            ## Remove by coordinate, boolean
            for ct in self.param['boundaries_excluded_boolean']:
                tol=1e-6
                operation=ct[0]
                coordinate=ct[1]
                value=ct[2]
                if len(ct)>3:
                    tol=ct[3]
                new_pool_runs=self.remove_by_boolean_condition (new_pool_runs, operation, coordinate, value,tol)
            
            pool_runs.extend(new_pool_runs)
            pool_phase=[np.random.random_sample()*2.0*np.pi for _ in range(len(pool_runs))]       
            pool_energies=[self.energy_0]*len(pool_runs)

            number_of_electrons,electrons_last_cycle=self.run_n_electrons_parallel (power, pool_runs, pool_phase, pool_energies, logfile_name, logtime_name)  

            total_electrons.append(number_of_electrons)
            final_electrons.append(electrons_last_cycle)
            power_partial.append(power)

        np.savetxt('generated_files/calculo_multipacting.txt', np.transpose([power_partial,total_electrons,final_electrons]), delimiter='\t') 
        if self.param['plot']:
            normalized_electrons=np.array(total_electrons)/self.param['electrons_seed']
            normalized_electrons_final=np.array(final_electrons)/self.param['electrons_seed']
            plt.plot (power_partial,normalized_electrons,'b-',label='Total')
            plt.plot (power_partial,normalized_electrons_final,'r-',label=f'Alive after {self.param["N_max_secondary_runs"]}')
            plt.xlabel ('Power (W)')
            plt.ylabel ('Normalized # of electrons')
            plt.title (self.param['plot_title'])
            plt.legend()
            plt.show ()

        #Mostrar tiempo total
        end_total = datetime.now()
        minutos_total = end_total.minute - start_total.minute + ((end_total.second - start_total.second)/60) + ((end_total.microsecond-start_total.microsecond)/60000000) + ((end_total.hour-start_total.hour)*60)
        print ("Total time:", format(minutos_total,".3f"), "min")

        #Guardar tiempo total
        logtime=open (self.logtime_name, 'a')
        if (minutos_total >= 1):
            logtime.write (f'Total execution time: {format(minutos_total,".3f")} min\n')
        else:
            segundos_total = minutos_total*60
            logtime.write (f'Total execution time: {format(segundos_total,".3f")} seg\n')
        logtime.close ()
    
def fowler_nordhaim_current_density (E, beta=100):
    ep=E*beta
    x=-6.65e10/ep
    jfn=0.0
    try:
        jfn=4e-5*math.pow (ep,2)
        jfn*=math.exp (x)
    except OverflowError:
        jfn=0.0
        
    return jfn

def face_normal (X):
    P1=X[0]
    P2=X[1]
    P3=X[2]
    BmA=P2-P1
    CmA=P3-P1
    N1=BmA[1]*CmA[2]-BmA[2]*CmA[1]
    N2=BmA[2]*CmA[0]-BmA[0]*CmA[2]
    N3=BmA[0]*CmA[1]-BmA[1]*CmA[0]
    mn=math.sqrt(N1*N1+N2*N2+N3*N3)
    N=np.array([N1,N2,N3])
    N=N/mn
    
    # print("BmA: {}\n".format(BmA))
    # print("CmA: {}\n".format(CmA))
    # print("N1: {}\n".format(N1))
    # print("N2: {}\n".format(N2))
    # print("N3: {}\n".format(N3))
    # print("mn: {}\n".format(mn))
    # print("N: {}\n".format(N))
    # sys.exit(0)
    return N
      
def run_1_electron (mpc, power, face, rf_phase, energy, keep=False):
    y=mpc.track_1_e (electron_energy=energy, power=power, phase=rf_phase, face_i=face, keep=keep)
    return y

