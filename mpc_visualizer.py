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


def plot_surface_mesh (bmesh):
    vedo_plotter=None
    if bmesh:
        vedo_plotter=plot (bmesh, color='gray', wireframe=True, interactive=False)

    return vedo_plotter


mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), 'coaxial_103mm_704MHz.mphtxt.h5', 'r')
hdf.read(mesh, "/mesh", False)
mesh_subdomains=MeshFunction("size_t", mesh, mesh.topology().dim())
hdf.read(mesh_subdomains, "/subdomains")
mesh_boundaries=MeshFunction("size_t", mesh, mesh.topology().dim()-1)
hdf.read(mesh_boundaries, "/boundaries")

bmesh=BoundaryMesh(mesh,'exterior')

trayectoria = []
with open('/home/alejandro/Desktop/Universidad/DIPC/5_mpc_cc_tests/build/generated_files/ultima_trayectoria.txt', 'r') as file:
    # Iterate over each line in the file
    for line in file.readlines():
        # Split the line into a list of strings, then map the float function onto this list to convert the strings to floats
        # Finally, convert the map object back into a list and append it to your data list
        trayectoria.append(list(map(float, line.split())))
        

energia_electron = []

# Open your file
with open('/home/alejandro/Desktop/Universidad/DIPC/5_mpc_cc_tests/build/generated_files/ultima_energia.txt', 'r') as file:
    # Iterate over each line in the file
    for line in file.readlines():
        # Append each line as a float to your list
        energia_electron.append(float(line.strip()))

vp=plot_surface_mesh(bmesh=bmesh)
vp+=pointcloud.Points(trayectoria, r=3, c='red')
vp.show(interactive=True)
# plt.plot (energia_electron)
# plt.ylabel ('electron energy (eV)')
# plt.show()

