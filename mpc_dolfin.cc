#include <string>
#include <cerrno>
#include <iostream>
#include <cmath>
#include <map>
#include <dolfin.h>
#include <MeshFunction.h>
#include <HDF5File.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cstdlib>

using namespace std;
using namespace dolfin;



class multipacting {
    public:
        string data_file;
        string mesh_file;
        Mesh mesh;
        BoundaryMesh bmesh;
        shared_ptr<MeshFunction<size_t>> mesh_subdomains;
        shared_ptr<MeshFunction<size_t>> mesh_boundaries;
        vector<double> mesh_center;
        BoundingBoxTree tree;
        BoundingBoxTree btree;

        bool data_ok;
        bool mesh_ok;
        Function campoEx;
        Function campoEy;
        Function campoEz;

        int N_ext;
        vector<vector<double>> lut_EX0;
        vector<double> lut_E0;
        vector<bool> lut_sense;
        map<string, pair<unsigned int, double>> closest_entity_dictionary;


        double RF_frequency; 
        double energy_0; 
        int N_cycles;
        double delta_t; 

        double e_m;
        double tol_distance;

        double angular_frequency;
        double electron_e_over_mc2;
        double c2;
        double electron_e_over_2mc;
        double electron_e_over_2m;

        vector<int> solid_domains;
        std::vector<size_t> domains_map;

        int max_workers;
        
        int N_runs_per_power;
        int N_max_secondary_runs;
        int macro;

        bool verbose;
        string logfile_name;
        string logtime_name;
        bool log;
        bool show;
        string plot_title;

        int randSeed;

        map<string, string> param;

        int N_cells;
        int N_elems;

        vector<double> RF_Power;

        void set_parameters_dictionary(map<string,string> param)
        {
            /*
            Recibe param, que es un diccionario con todos los parámetros
            Uno a uno va inicializando los atributos de la clase según lo que encuentra en param
            */
            this->param = param;

            this->log     = (param["log"] == "True");
            this->verbose = (param["verbose"] == "True");

            this->N_runs_per_power     = stoi(param["electrons_seed"]);
            this->N_max_secondary_runs = stoi(param["N_max_secondary_runs"]);
            this->randSeed             = stoi(param["random_seed"]);
            
            if (param["RF_power"].find("range") != string::npos){ // Significa que 'range' se encuentra en el parámetro
                string ps = param["RF_power"];
                ps = ps.substr(ps.find("(")+1, ps.find(")") - ps.find("("));
                // Ahora 'ps' tiene el valor de incio, el del final y el incremento
                // Es decir: (x1,x2,x3)
                string ps2 = ps.substr(ps.find(","),ps.length()-ps.find(",")-1); // ps2 = x2,x3
                float x1 = stof(ps.substr(0,ps.find(",")));
                float x2 = stof(ps2.substr(0,ps2.find(",")));
                float x3 = stof(ps2.substr(ps2.find(",")+1,ps2.length()-1));

                // Con esos valores se crea un array
                // this->RF_Power.resize(((x1+x2)/x3)+1);
                for (float i = x1; i < x2; i+=x3) this->RF_Power.push_back(i);

            } else {
                float x1 = stof(param["RF_power"]);
                this->RF_Power.push_back(x1);
            }

            this->RF_frequency = stod(param["RF_frequency"]);
            this->angular_frequency = 2*M_PI*this->RF_frequency;
            this->delta_t = stod(param["delta_t"]);
            this->N_cycles = stoi(param["N_cycles"]);

            if (param["plot_title"].length() > 0) this->plot_title = param["plot_title"];
            if (param.find("comsol_solid_domains") != param.end()) this->set_solid_domains_COMSOL(param["comsol_solid_domains"]);
        }

        int get_N_surface_elements() {return N_ext;} // Esto es necesario? Si fuese un parámetro privado entiendo que sí, pero si es público?
        
        bool read_mesh_file(string mesh_file = ""){
            if (mesh_file != "") this->mesh_file = mesh_file;
            // Dice que esto es EXTREMADAMENTE importante, pero no se define 'parameters' en ningún lado del código original
            //parameters["reorder_dofs_serial"] = false;
            
            mesh = Mesh();
            HDF5File hdf(MPI_COMM_WORLD, this->mesh_file, "r");

            mesh_subdomains = make_shared<MeshFunction<size_t>>(mesh, mesh.topology().dim());
            hdf.read(mesh_subdomains, "/subdomains");
            
            mesh_boundaries = make_shared<MeshFunction<size_t>>(mesh, mesh.topology().dim()-1);
            hdf.read(mesh_boundaries, "/boundaries");

            // Mesh analysis
            N_cells = mesh.num_cells(); // tetras

            bmesh = BoundaryMesh(mesh, "exterior");
            N_ext = bmesh.num_cells();
            mesh_center = mesh_center_point(bmesh);

            tree = BoundingBoxTree();
            tree.build(mesh,3);

            btree = BoundingBoxTree();
            btree.build(bmesh,2);

            N_elems = N_ext;

            domains_map.assign(mesh_boundaries->values(), mesh_boundaries->values() + mesh_boundaries->size());

            for (int i = 0; i < lut_sense.size(); i++) lut_sense[i] = false;

            mesh_ok = true;

            // PENDIENTE: añadir el catch por si no se puede abrir bien el file HDF5, que devuelve false
            return true;
        }

        vector<double> mesh_center_point(Mesh mesh) {
            vector<double> mc = mesh.coordinates();

            double minx = std::numeric_limits<double>::max();
            double maxx = std::numeric_limits<double>::lowest();;
            double miny = std::numeric_limits<double>::max();;
            double maxy = std::numeric_limits<double>::lowest();;
            double minz = std::numeric_limits<double>::max();;
            double maxz = std::numeric_limits<double>::lowest();;

            for (size_t i = 0; i < mc.size(); i += 3) {
                double x = mc[i];
                double y = mc[i + 1];
                double z = mc[i + 2];
                minx = min(minx, x);
                miny = min(miny, y);
                minz = min(minz, z);
                maxx = max(maxx, x);
                maxy = max(maxy, y);
                maxz = max(maxz, z);
            }

            double median_x = (maxx + minx) / 2.0;
            double median_y = (maxy + miny) / 2.0;
            double median_z = (maxz + minz) / 2.0;

            double Xc = median_x;
            double Yc = median_y;
            double Zc = median_z;

            return {Xc, Yc, Zc};
        }


        void set_solid_domains_COMSOL(string ld) {

            // Ya que ld es un string de la forma [x y ... z]
            stringstream ss(ld);
            char openBracket, closeBracket;
            int domain;
            
            ss >> openBracket; // Leer el corchete de apertura '['

            this->solid_domains.clear();
            while (ss >> domain) {
                solid_domains.push_back(domain + 1);
            }
            
            ss.clear();
            ss >> closeBracket; // Leer el corchete de cierre ']'
        }

        map<size_t,size_t> domain_histogram(bool show = false){
            std::vector<size_t> md;
            md.assign(mesh_boundaries->values(), mesh_boundaries->values() + mesh_boundaries->size());
            map<size_t,size_t> dh;

            for (const auto& n : md)
            {
                if (dh.find(n) != dh.end()) dh[n] += 1;
                else    dh[n] = 1;
            }

            if (show) for (const auto& n : dh) std::cout << n.first << ": " << n.second << std::endl;

            return dh;
        }

        bool read_field_data(bool build_lookup_table = true){


            std::ifstream file(data_file);
            if (!file.is_open()) {
                std::cerr << "Failed to open the data file." << std::endl;
                return false;
            }

            // Skip header lines
            for (int i = 0; i < 9; ++i) {
                std::string line;
                std::getline(file, line);
            }

            std::string line;
            vector<double> EX;
            vector<double> EY;
            vector<double> EZ;
            vector<double> X;
            vector<double> Y;
            vector<double> Z;
            
            // Read E B field and process
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                double x, y, z, ex_real, ex_imag, ey_real, ey_imag, ez_real, ez_imag;
                char delimiter;
                if (iss >> x >> delimiter >> y >> delimiter >> z >> delimiter >> ex_real >> delimiter >> ex_imag
                        >> delimiter >> ey_real >> delimiter >> ey_imag >> delimiter >> ez_real >> delimiter >> ez_imag) {
                    // Real part
                    EX.push_back(ex_real);
                    EY.push_back(ey_real);
                    EZ.push_back(ez_real);
                    X.push_back(x);
                    Y.push_back(y);
                    Z.push_back(z);
                }
            }

            // FEniCS black magic
            // I create a finite element function space on the mesh I have read.
            // field data will be interpolated (projected) into that vector space
            shared_ptr<FiniteElement> element = make_shared<FiniteElement>("CG", mesh.topology().dim(), 1);
            shared_ptr<FunctionSpace> V = make_shared<FunctionSpace>(mesh, element, 1);

            // Defino las funciones para el campo dentro del espacio de funciones V
            auto Fez = Function(V);
            auto Fex = Function(V);
            auto Fey = Function(V);

            Fez.vector()->set_local(EZ);
            Fex.vector()->set_local(EX);
            Fey.vector()->set_local(EY);

            // PENDIENTE: proyectar

            // Function VFez(V);  // Esta será la proyección de Fez en el espacio V

            // // Define la ecuación variacional para la proyección
            // ufl::TrialFunction v(V); // Función que cumple el contorno (?) 
            // ufl::TestFunction u(V);  // Función que se resuelve de 'manera débil', 
            // ufl::Form a = ufl::inner(u, v) * ufl::dx; // Se define la forma bilineal según la sintáxis de ufl porque es más sencilla 
            // ufl::Form L = ufl::inner(Fez, v) * ufl::dx; // Lo mismo

            // // Resuelve la ecuación
            // dolfin::solve(a == L, VFez);
            // // En resumen se busca una función v tal que la integral de v*u sea igual a la integral de v*Fez

            // this->campoEx = campoEx;

            if (!build_lookup_table) return true;

            std::cout << N_ext << std::endl;

            for(int i=0; i<N_ext; i++)
            {
                auto facet_i = Face(bmesh,i);
                auto mp = facet_i.midpoint();
                auto X0 = mp.coordinates();

                // auto EX0x = campoEx(X0);
                // auto EX0y = campoEy(X0);
                // auto EX0z = campoEz(X0);

                vector<double> EX0;
                // EX0.push_back(EX0x);
                // EX0.push_back(EX0y);
                // EX0.push_back(EX0z);

                // E0 = sqrt(sum(square(EX0)))
                double E0 = sqrt(((EX0[0] * EX0[0]) + (EX0[1] * EX0[1]) + (EX0[2] * EX0[2])));
                lut_EX0.push_back(EX0);
                lut_E0.push_back(E0);
            }

            return true;

        }

        bool read_from_data_files(string mesh_file, string data_file){
            this->data_file = data_file;
            this->mesh_file = mesh_file;

            bool mesh_ok = read_mesh_file();
            bool field_ok = read_field_data();

            if (!(mesh_ok && field_ok)) return false;

            if (verbose)
            {
                std::cout << "Data file: " << data_file << std::endl;
                std::cout << "Mesh file: " << mesh_file << std::endl;
                std::cout << "Data read. Num surface elems:  " << N_elems << ", Num volume elems: " << N_cells << std::endl;
            }

            return true;
        }

        bool read_input_files(){return read_from_data_files(param["mesh_file"],param["data_file"]);}

        void plot_surface_mesh(){
            // El siguiente código está sacado de https://fenicsproject.org/olddocs/dolfin/1.3.0/cpp/programmers-reference/plot/VTKPlotter.html
            // Pero no consigo que haga el include de VTKPlotter
            // vtk::VTKPlotter plotter(bmesh);
            // plotter.plot();

            // Resulta que ya no se puede:
            // Remove VTK plotting backend. Plotting is no longer available from 
            // the C++ interface. Basic plotting is available using matplotlib 
            // and x3dom backends via the plot() free function in the Python 
            // interface. Users are advised to move to e.g. Paraview for more 
            // demanding plotting needs.

            // Tengo una alternativa larga y odiosa escrita en plot_surface_mesh.cc
            // pero, igual no merece y dejamos esta funcionalidad de lado?

            // Se puede hacer así:
            // vtk::VTKFile file("bmesh.pvd");
            // file << bmesh;
            // Y luego visualizarlo externamente usando paraview
        }

        std::pair<unsigned int, double> closest_entity(vector<double> values){
            
            std::stringstream sstream;
            sstream << std::fixed << std::setprecision(4) << values[0] << "_" << values[1] << "_" << values[2];
            std::string strx = sstream.str();

            std::pair<unsigned int, double> D;  

            if (closest_entity_dictionary.find(strx) != closest_entity_dictionary.end()){
                D = closest_entity_dictionary[strx];
            } else {
                dolfin::Point Xp(X[0],X[1],X[2]);
                D = tree.compute_closest_entity(Xp);
                closest_entity_dictionary[strx] = D;
            }

            return D;
            
        }

        bool point_inside_mesh(vector<double> X){
            Point p(X[0],X[1],X[2]);
            auto ent = tree.compute_first_entity_collision(p);
            if (ent >= N_cells) return false;

            if (!solid_domains.empty())
            {
                unsigned int domain_index = domains_map[ent];
                if (find(solid_domains.begin(),solid_domains.end(),domain_index) != solid_domains.end()) return false;
            }

            return true;


        }

        std::vector<std::vector<double>> get_initial_conditions_face(int face_i)
        {
            // Como no hay numpy en C++, asumo que en lugar de venir en formato
            // ([0,1,3],[3,5,6]) viene en formato (0,1,3,3,5,6)
            // Para cada cell, vienen 3 unsigned int que identifican los vértices que lo componen
            // Asumo que son triángulos
            std::vector<unsigned int> cells = bmesh.cells();
            vector<unsigned int> nodes;
            nodes.push_back(cells[face_i*3]);
            nodes.push_back(cells[face_i*3+1]);
            nodes.push_back(cells[face_i*3+2]);

            // Lo mismo ocurre aquí, asumo que en 
            // lugar de ([1,2,3],[1,2,3]) viene como (1,2,3,1,2,3)
            // Ya que estamos en 3D, las coordenadas de cada vértice son 3 doubles
            std::vector<double> coords = bmesh.coordinates();
            vector<double> X;
            for (int i=0;i<3;i++) 
            for (int j=0;j<3;j++)
            X.push_back(coords[nodes[i]*3+j]);
            
            dolfin::Face facet_i = dolfin::Face(bmesh,face_i);
            dolfin::Point mp = facet_i.midpoint();
            std::vector<double> Nv = face_normal(X); // La normal

            // Element sense (normal pointing inward or outward)
            std::vector sense_factor = lut_sense[face_i];
            if (!sense_factor)
            {
                // Calcula la distancia desde el centro del mesh al centro de la celda face_i
                std::vector<double> Xc = mesh_center;
                double dcm = sqrt(((mp[0]-Xc[0])**2)+((mp[1]-Xc[1])**2)+((mp[2]-Xc[2])**2));

                // Avanzo 0.1*distancia en la dirección de la normal
                double Xp = mp[0] + (dcm*0.1*Nv[0]);
                double Yp = mp[1] + (dcm*0.1*Nv[1]);
                double Zp = mp[2] + (dcm*0.1*Nv[2]);

                std::vector<double> XYZp = {Xp,Yp,Zp};

                std::pair<unsigned int,double> dcm_pair = closest_entity(XYZp);
                std::double dcm_d = dcm_pair.second;

                sense_factor = true;
                if (dcm_d > 0.0) sense_factor = false; // Asumo que compute_closest_entity de BoundingBoxTree.h funciona igual que en la API de python
                
                lut_sense[face_i] = sense_factor;
            }

            // Tracking, starting conditions
            std::vector<double> X0 = {mp[0],mp[1],mp[2]};
            std::vector<double> U0;
            // PENDIENTE: Se puede hacer esto más elegante seguro
            for (int i=0;i<3;i++){
                if (sense_factor == false) U0.push_back((-1)*(Nv[i]));
                else U0.push_back(Nv[i]);
            }
            std::vector<double> EX0 = lut_EX0[face_i];
            std::vector<std::vector<double>> result(3,std::vector<double>(3));
            result[0] = X0;
            result[1] = U0;
            result[2] = EX0;

            return result;
        }

        void track_1_e(double electron_energy_in = nullptr, double power = 1.0, double phase = 0.0, int face_i_in = nullptr, bool keep_in = false, bool show = false, dolfin::Point starting_point_in = nullptr){
            /* Runs the tracking of 1 electron in the problem geometry.
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
            */
           bool magnetic_field = param["magnetic_field_on"];
           double field_factor = sqrt(power);
           double electron_energy = electron_energy_in;
           int face_i = face_i_in;
           dolfin::Point starting_point = dolfin::point(starting_point_in);  // Starting_point no es un punto!!! Creo que es unas coordenadas 3D y una velocidad o energía
           bool keep = keep_in;

           trayectoria;
           collision;
           energy_collision;
           phase_collision;
           max_gamma=0;
           V0 = nullPtr;
           U0 = nullPtr;

           if (electron_energy == nulPtr) electron_energy = energy_0; 
           if (show) keep = true;

           if (log){
               std::ofstream logfile;
               logfile.open(logfile_name,'a');
               logfile << "Call track_1_e starting from face=" << face_i
           }

            std::vector<double> X0;
            std::vector<double> U0;
            std::vector<double> EX0;

           if (starting_point == nullPtr){
               if (face_i == nullPtr) {
                   srand(time(0));
                   face_i = rand() % (N_ext + 1);
               }
               auto results = get_initial_conditions_face(face_i);
               X0 = results[0];
               U0 = results[0];
               EX0 = results[0];
           } else {
               X0 = starting_point[0];
           }
        }

        // void total_secondary_electrons();
        // void secondary_electron_yield();
        // void efn_emmision();
        // void remove_by_coordinate_value();
        // void remove_by_boolean_condition();
        // void probability_of_emmision();
        // void run_1_electron();
        // void run_n_electrons_parallel();
        // void run();






};




double fowler_nordhaim_current_density(double E, int beta = 100)
{
    double ep = E*beta;
    double x = -6.65e10/ep;
    double jfn = 0.0;

    jfn  = 4e-5*pow(ep,2);
    jfn *= exp(x);
    
    if (errno == ERANGE)
        jfn = 0.0;
    
    return jfn;
} 

std::vector<double> face_normal(std::vector<double> X){
    // X es un vector de double que contiene las coordenadas 3D de los 3 vértices de un nodo
    // X = (1,2,3,4,5,6,7,8,9) = [(1,2,3),(4,5,6),(7,8,9)] por ejemplo
    int i;

    std::vector<double> P1;
    std::vector<double> P2;
    std::vector<double> P3;
    for (i=0;i<3;i++){
        P1.push_back(X[i]);
        P2.push_back(X[i+3]);
        P3.push_back(X[i+6]);
    }

    std::vector<double> BmA;
    std::vector<double> CmA;

    for (i=0;i<3;i++){
        BmA.push_back(P2[i]-P1[i]);
        CmA.push_back(P3[i]-P1[i]);
    }

    double N1=BmA[1]*CmA[2]-BmA[2]*CmA[1];
    double N2=BmA[2]*CmA[0]-BmA[0]*CmA[2];
    double N3=BmA[0]*CmA[1]-BmA[1]*CmA[0];

    double mn = sqrt(N1*N1 + N2*N2 + N3*N3);
    std::vector<double> N;
    N.push_back(N1/mn);
    N.push_back(N2/mn);
    N.push_back(N3/mn);

    return N;

}


void run_1_electron(multipacting mpc)
{
    // return mpc.track_1_e(); 
}
