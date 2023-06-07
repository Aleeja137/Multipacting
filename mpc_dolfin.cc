#include <string>
#include <cerrno>
#include <iostream>
#include <cmath>
#include<map>
using namespace std;

class multipacting {
    public:
        string data_file;
        string mesh_file;
        // Mesh() mesh
        // BoundaryMesh() bmesh
        // MeshFunction() mesh_boundaries
        // MeshFunction() mesh_subdomains
        float mesh_center[3];
        // BuildingBoxTree() tree
        // BuildingBoxTree() btree

        bool data_ok;
        bool mesh_ok;
        // project() campoEx
        // project() campoEy
        // project() campoEz

        int N_ext;
        // array ?? lut_EX0
        // array ?? lut_E0
        // int lut_sense[]; // Como no sé el tamaño, uso std::vector<int> ?
        // map(dictionary) closest_entity_dictionary

        float RF_frequency; 
        float energy_0; 
        int N_cycles;
        float delta_t; 

        double e_m;
        double tol_distance;

        double angular_frequency;
        double electron_e_over_mc2;
        double c2;
        double electron_e_over_2mc;
        double electron_e_over_2m;

        // ?? solid_domains
        // ?? domains_map

        int max_workers;
        
        map<string,string> param;

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
                // Ahora ps tiene el valor de incio, el del final y el incremento
                // Es decir: (x1,x2,x3)
                string ps2 = ps.substr(ps.find(","),ps.length()-ps.find(",")-1); // ps2 = x2,x3
                float x1 = stof(ps.substr(0,ps.find(",")));
                float x2 = stof(ps2.substr(0,ps2.find(",")));
                float x3 = stof(ps2.substr(ps2.find(",")+1,ps2.length()-1));
                // Con esos valores se crea un array numpy 
                // this.RF_power = np.arange(x1,x2,x3);
                // this->param["RF_power"] = this.RF_power;
            } else {
                float x1 = stof(param["RF_power"]);
                // this->param["RF_power"] = np.array(x1); // x1 debe ser un array (?) 
            }

            this->RF_frequency = stof(param["RF_frequency"]);
            this->angular_frequency = 2*M_PI*this->RF_frequency;
            this->delta_t = stof(param["delta_t"]);
            this->N_cycles = stoi(param["N_cycles"]);

            if (param["plot_title"].length() > 0) this->plot_title = param["plot_title"];
            if (param.find("comsol_solid_domains") != param.end()) this->set_solid_domains_COMSOL(param["comsol_solid_domains"]);
        }
        int get_N_surface_elements() {return N_ext;} // Esto es necesario? Si fuese un parámetro privado entiendo que sí, pero si es público?
        void read_mesh_file();
        void mesh_center_point();
        void set_solid_domains_COMSOL(string ld);
        void domain_histogram();
        void read_field_data();
        void read_from_data_files();
        void read_input_files();
        void plot_surface_mesh();
        void closest_entity();
        void point_inside_mesh();
        void get_initial_conditions_face();
        void track_1_e();
        void total_secondary_electrons();
        void secondary_electron_yield();
        void efn_emmision();
        void remove_by_coordinate_value();
        void remove_by_boolean_condition();
        void probability_of_emmision();
        void run_1_electron();
        void run_n_electrons_parallel();
        void run();






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