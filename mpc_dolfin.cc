#include <string>
#include <cerrno>
#include <iostream>
#include <cmath>
#include <map>
#include <dolfin.h>
#include <MeshFunction.h>
#include <dolfin/io/HDF5File.h>
#include <fstream>

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
        
        void read_mesh_file(string mesh_file = ""){
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

            // PENDIENTE: añadir el catch por si no se puede abrir bien el file HDF5
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
            vector<complex<double>> EX;
            vector<complex<double>> EY;
            vector<complex<double>> EZ;
            vector<double> X;
            vector<double> Y;
            vector<double> Z;
            
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                double x, y, z, ex_real, ex_imag, ey_real, ey_imag, ez_real, ez_imag;
                char delimiter;
                if (iss >> x >> delimiter >> y >> delimiter >> z >> delimiter >> ex_real >> delimiter >> ex_imag
                        >> delimiter >> ey_real >> delimiter >> ey_imag >> delimiter >> ez_real >> delimiter >> ez_imag) {
                    std::complex<double> ex(ex_real, ex_imag);
                    std::complex<double> ey(ey_real, ey_imag);
                    std::complex<double> ez(ez_real, ez_imag);
                    EX.push_back(ex);
                    EY.push_back(ey);
                    EZ.push_back(ez);
                    X.push_back(x);
                    Y.push_back(y);
                    Z.push_back(z);
                }
            }

        }

        // void read_from_data_files();
        // void read_input_files();
        // void plot_surface_mesh();
        // void closest_entity();
        // void point_inside_mesh();
        // void get_initial_conditions_face();
        // void track_1_e();
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

void face_normal();
void run_1_electron(multipacting mpc)
{
    // return mpc.track_1_e(); 
}
