#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <dolfin.h>
#include <dolfin/io/HDF5File.h>
#include <mpi.h>
#include "MyLagrange.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <boost/units/systems/si/codata/electromagnetic_constants.hpp>
#include <boost/units/systems/si/codata/electron_constants.hpp>
#include <boost/units/systems/si/codata/universal_constants.hpp>
#include <boost/math/constants/constants.hpp>
#include <typeinfo>

// Class variables
bool log_exec = false;
bool debug = false;
bool plot = false;
bool verbose = false;
bool magnetic_field_on;
bool mesh_ok = false;
bool build_lookup_table;
bool show = false;

int parallel = 1;
int electrons_seed = 100;
int electrons_source = 1;
int N_max_secondary_runs = 15;
int simulation_type = 1;
int random_seed = -1;
int N_runs_per_power;
int N_cycles;
int N_cells;
int N_elems;
int N_ext;

double delta_t = 1e-10;
double RF_frequency = 2.7e9;
double version;
double angular_frequency;
double energy_0 = 1.0;
double elementary_charge  = boost::units::si::constants::codata::e   / boost::units::si::coulomb;
double electron_mass      = boost::units::si::constants::codata::m_e / boost::units::si::kilogram;
double speed_light_vacuum = boost::units::si::constants::codata::c   / boost::units::si::meter_per_second;
double electron_e_over_mc2 = elementary_charge / (electron_mass*speed_light_vacuum*speed_light_vacuum);
double electron_e_over_2mc = elementary_charge / (2.0*electron_mass*speed_light_vacuum);
double electron_e_over_2m  = elementary_charge / (2.0*electron_mass);

std::string data_file  = "";
std::string mesh_file  = "";
std::string plot_title = "";
std::string file_mode = "r";
std::string logfile_name = "";
std::string logtime_name = "";

std::vector<int> comsol_solid_domains;
std::vector<int> solid_domains;
std::vector<std::vector<double>> boundaries_excluded_boolean; 
std::vector<double> mesh_center;
std::vector<double> RF_power = {1000};
std::vector<double> lut_E0;

std::vector<Eigen::Vector3d> lut_EX0;

std::size_t* domains_map = nullptr;
std::size_t domains_map_size;

std::vector<bool> lut_sense;

std::map<std::string, double> closest_entity_dictionary;

// Parameter list
std::vector<std::string> boolean_arguments = {"log","debug","plot","verbose","magnetic_field_on"};
std::vector<std::string> int_arguments = {"parallel","electrons_seed","electrons_source","simulation_type","N_max_secondary_runs","N_cycles","random_seed"};
std::vector<std::string> double_arguments = {"delta_t","RF_frequency"};
std::vector<std::string> eval_arguments = {"comsol_solid_domains","X0"};
std::vector<std::string> string_arguments = {"electric_data_file","magnetic_data_file","data_file","mesh_file","RF_power","plot_title"};
std::vector<std::string> boundary_arguments = {"boundaries_excluded_boolean"};

// Map to read parameters
std::map<std::string,std::string> p;

// Dolfin variables
std::shared_ptr<dolfin::Mesh> mesh;
std::shared_ptr<dolfin::MeshFunction<std::size_t>> mesh_subdomains;
std::shared_ptr<dolfin::MeshFunction<std::size_t>> mesh_boundaries;
std::shared_ptr<dolfin::BoundaryMesh> bmesh;
std::shared_ptr<dolfin::BoundingBoxTree> tree;
std::shared_ptr<dolfin::BoundingBoxTree> btree;

std::shared_ptr<dolfin::Function> campoEx;
std::shared_ptr<dolfin::Function> campoEy;
std::shared_ptr<dolfin::Function> campoEz;



// _____________ Funciones para check_______________________//

void print_info(){
    // Asumiendo que tus variables son miembros de una clase y son accesibles desde aquí

    std::cout << "mesh_file: " << mesh_file << ", type: " << typeid(mesh_file).name() << std::endl;
    std::cout << "N_cells: " << N_cells << ", type: " << typeid(N_cells).name() << std::endl;
    std::cout << "N_ext: " << N_ext << ", type: " << typeid(N_ext).name() << std::endl;
    std::cout << "N_elems: " << N_elems << ", type: " << typeid(N_elems).name() << std::endl;
    std::cout << "mesh_ok: " << mesh_ok << ", type: " << typeid(mesh_ok).name() << std::endl;

    std::cout << "domains_map size: " << mesh_subdomains->size() << std::endl;
    bool check = true;
    for (int i=0;i<domains_map_size;i++)if (domains_map[i]!=2)check=false;
    std::cout << "Check if they are all value '2': " << check << std::endl;

    check=true;
    std::cout << "print valu true: " << check << std::endl;
    
    std::cout << "lut_sense size: " << lut_sense.size() << std::endl;
    check = true;
    for (int i=0;i<lut_sense.size();i++)if (lut_sense[i]!=false)check=false;
    std::cout << "Check if they are all false: " << check << std::endl;

    std::cout << "Type of mesh: " << typeid(mesh).name() << ", geometry: " << mesh->geometry().dim() << ", n_vertices: " << mesh->num_vertices() << ", n_cells: " << mesh->num_cells() << ", n_facets: " << mesh->num_facets() << std::endl;

    std::cout << "mesh_subdomains type: " << typeid(mesh_subdomains).name() << ", n_values: " << mesh_subdomains->size() << ", each element: " << std::endl;
    auto values = mesh_subdomains->values();

    std::cout << "mesh_boundaries type: " << typeid(mesh_boundaries).name() << ", n_values: " << mesh_boundaries->size() << ", each element: " << std::endl;
    
    std::cout << "Mesh center: \n" << std::endl;
    for(auto element : mesh_center){
        std::cout << element << std::endl;
    }

    std::cout << "bmesh type: " << typeid(bmesh).name() << ", n_vertices: " << bmesh->num_vertices() << ", n_cells: " << bmesh->num_cells() << std::endl;
}

void print_all_variables(){ // Imprime TODAS las variables para saber el estado

    std::cout << "log_exec: " << log_exec << std::endl;
    std::cout << "debug: " << debug << std::endl;
    std::cout << "plot: " << plot << std::endl;
    std::cout << "verbose: " << verbose << std::endl;
    std::cout << "magnetic_field_on: " << magnetic_field_on << std::endl;
    std::cout << "N_runs_per_power: " << N_runs_per_power << std::endl;
    std::cout << "N_cycles: " << N_cycles << std::endl;
    std::cout << "N_cells: " << N_cells << std::endl;
    std::cout << "N_elems: " << N_elems << std::endl;

    std::cout << "parallel: " << parallel << std::endl;
    std::cout << "electrons_seed: " << electrons_seed << std::endl;
    std::cout << "electrons_source: " << electrons_source << std::endl;
    std::cout << "N_max_secondary_runs: " << N_max_secondary_runs << std::endl;
    std::cout << "simulation_type: " << simulation_type << std::endl;
    std::cout << "random_seed: " << random_seed << std::endl;

    std::cout << "delta_t: " << delta_t << std::endl;
    std::cout << "RF_frequency: " << RF_frequency << std::endl;
    std::cout << "angular_frequency: " << angular_frequency << std::endl;

    std::cout << "--- RF_power ---" << std::endl;
    for (auto element : RF_power) std::cout << element << " ";
    std::cout << std::endl;

    std::cout << "version: " << version << std::endl;

    std::cout << "data_file: " << data_file << std::endl;
    std::cout << "mesh_file: " << mesh_file << std::endl;
    std::cout << "plot_title: " << plot_title << std::endl;

    std::cout << "--- comsol_solid_domains ---" << std::endl;
    for (auto element : comsol_solid_domains) std::cout << element << " ";
    std::cout << std::endl;

    std::cout << "--- boundaries_excluded_boolean ---" << std::endl;
    for (auto element : boundaries_excluded_boolean) {
        for (auto value : element){
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;    
}

void print_map(std::map<std::string,std::string> map_in){
    for (auto element : map_in)
    {
        std::cout << element.first << " : " << element.second << std::endl;
    }
}

void comprobar_compilacion_dolfin(){
    if (dolfin::has_hdf5) std::cout << "Dolfin has HDF5" << std::endl;
    else std::cout << "Dolfin DOES NOT HAVE HDF5" << std::endl;

    if (dolfin::has_hdf5_parallel) std::cout << "Dolfin has HDF5 parallel" << std::endl;
    else std::cout << "Dolfin DOES NOT HAVE HDF5 parallel" << std::endl;

    if (dolfin::has_mpi) std::cout << "Dolfin has mpi" << std::endl;
    else std::cout << "Dolfin DOES NOT HAVE mpi" << std::endl;

    if (dolfin::has_openmp) std::cout << "Dolfin has openmp" << std::endl;
    else std::cout << "Dolfin DOES NOT HAVE openmp" << std::endl;

    if (dolfin::has_petsc) std::cout << "Dolfin has petsc" << std::endl;
    else std::cout << "Dolfin DOES NOT HAVE petsc" << std::endl;

    if (dolfin::has_slepc) std::cout << "Dolfin has slepc" << std::endl;
    else std::cout << "Dolfin DOES NOT HAVE slepc" << std::endl;
}

// _____________ Funciones que se usan______________________//

std::vector<double> face_normal(std::vector<std::vector<double>> X){
    std::vector<double> P1 = X[0];
    std::vector<double> P2 = X[1];
    std::vector<double> P3 = X[2];

    std::vector<double> BmA, CmA, N;
    double N1, N2, N3, mn;

    for (int i=0;i<3;i++)
    {
        BmA.push_back(P2[i]-P1[i]);
        CmA.push_back(P3[i]-P1[i]);
    }

    N1 = BmA[1]*CmA[2] - BmA[2]*CmA[1];
    N2 = BmA[2]*CmA[0] - BmA[0]*CmA[2];
    N3 = BmA[0]*CmA[1] - BmA[1]*CmA[0];

    mn = std::sqrt((N1*N1)+(N2*N2)+(N3*N3));

    N.push_back(N1/mn);
    N.push_back(N2/mn);
    N.push_back(N3/mn);

    return N;
}

// Esta functión es mía, devuelve en una lista campoEx([x,y,z]),campoEy([x,y,z]),campoEz([x,y,z])
// Podría ahorrarmela si empiezo a usar eigen::vector3d para el álgebra
std::vector<double> evaluate_campo_from_point(double x, double y, double z){

    Eigen::VectorXd point(3);
    point << x,y,z;
    Eigen::Ref<const Eigen::VectorXd> ref_point = point;

    Eigen::VectorXd res_ex(1);
    Eigen::Ref<Eigen::VectorXd> ref_res_ex = res_ex;
    campoEx->eval(ref_res_ex,ref_point);

    Eigen::VectorXd res_ey(1);
    Eigen::Ref<Eigen::VectorXd> ref_res_ey = res_ey;
    campoEy->eval(ref_res_ey,ref_point);

    Eigen::VectorXd res_ez(1);
    Eigen::Ref<Eigen::VectorXd> ref_res_ez = res_ez;
    campoEz->eval(ref_res_ez,ref_point);

    Eigen::Vector3d values(ref_res_ex[0],ref_res_ey[0],ref_res_ez[0]);

    std::vector<double> result(values.data(),values.data() + values.size());

    return result;
}

bool point_inside_mesh(std::vector<double> X){
    dolfin::Point point(X[0], X[1], X[2]);
    auto ent = tree->compute_first_entity_collision(point);
    if (ent >= N_cells) return false;
    if (!solid_domains.empty())
    {
        auto domain_index = domains_map[ent];
        if (std::find(solid_domains.begin(), solid_domains.end(),domain_index) != solid_domains.end()) return false; // Point inside a solid domain
    }
    return true;
}

// Get a parameter and its argument from a line like 'param=arg#comment'
std::pair<std::string,std::string> get_param_arg(std::string line)
{
    if (line.empty() || line.find_first_not_of(' ') == std::string::npos) return std::make_pair("","");
    else if (line[0] == '#') return std::make_pair("","");
    else
    {
        std::string param;
        std::string argument;
        std::size_t index_equal = line.find('=');

        param    = line.substr(0,index_equal);
        argument = line.substr(index_equal+1);

        std::size_t index_comment = argument.find('#');

        argument = argument.substr(0, index_comment);

        return std::make_pair(param,argument);
    }
}

void set_solid_domains_COMSOL(std::string ld){
    std::string aux;
    for (char c : ld) if (std::isdigit(c)) 
    {
        aux = c;
        solid_domains.push_back(std::stoi(aux)+1);
    }
    // for (auto element : solid_domains) std::cout << element << " ";
    // std::cout << std::endl;
}

void set_parameters_dictionary(){
    data_file = "data/" + p["data_file"];
    mesh_file = "data/" + p["mesh_file"];

    log_exec = p["log"] == "True";
    verbose = p["verbose"] == "True";

    N_runs_per_power     = stoi(p["electrons_seed"]);
    N_max_secondary_runs = stoi(p["N_max_secondary_runs"]);
    random_seed          = stoi(p["random_seed"]);
    N_cycles             = stoi(p["N_cycles"]);
    simulation_type      = stoi(p["simulation_type"]);

    if (p["RF_power"].find("range") != std::string::npos){ // Significa que 'range' se encuentra en el parámetro
        
        std::string ps = p["RF_power"];
        std::size_t index_start = ps.find('(');
        std::size_t index_end = ps.find(')');
        std::string ps_limpio = ps.substr(index_start+1, index_end - index_start - 1);

        // std::cout << ps_limpio << std::endl;
        
        std::size_t index_coma = ps_limpio.find(',');
        double x1 = stod(ps_limpio.substr(0,index_coma));

        std::string ps_limpio_2 = ps_limpio.substr(index_coma+1);
        index_coma = ps_limpio_2.find(',');
        double x2 = stod(ps_limpio_2.substr(0,index_coma));
        double x3 = stod(ps_limpio_2.substr(index_coma+1));
        // std::cout << x1 << "," << x2 << "," << x3 << std::endl;
        RF_power.clear();
        for (double i = x1; i < x2; i+=x3) RF_power.push_back(i);

    } else {
        double x1 = stod(p["RF_power"]);
        RF_power.clear();
        RF_power.push_back(x1);
    }

    RF_frequency = stod(p["RF_frequency"]);
    angular_frequency = 2*M_PI*RF_frequency;
    delta_t = stod(p["delta_t"]);

    if (p.find("plot_title")!=p.end()) plot_title = p["plot_title"];
    if (p.find("comsol_solid_domains")!=p.end()) set_solid_domains_COMSOL(p["comsol_solid_domains"]);
    if (p.find("boundaries_excluded_boolean")!=p.end()) 
    {
        std::string input_str = p["boundaries_excluded_boolean"];
        std::vector<double> temp_list;
        std::size_t pos = 0;
        double value;
        while (pos < input_str.length())
        {
            try
            {
                std::size_t new_pos;
                value = stod(input_str.substr(pos),&new_pos);
                pos += new_pos;
                temp_list.push_back(value);
                if (temp_list.size() == 4)
                {
                    boundaries_excluded_boolean.push_back(temp_list);
                    temp_list.clear();
                }
            }
            catch(const std::exception& e) {pos++;}
        }
    }
}

std::vector<double> mesh_center_point(std::shared_ptr<dolfin::BoundaryMesh> mesh_in){
    auto mc = mesh_in->coordinates();
    double minx=0; double maxx=0;
    double miny=0; double maxy=0;
    double minz=0; double maxz=0;

    double x, y, z, median_x, median_y, median_z;

    for (int i = 0;i<mc.size();i+=3){
        x = mc[i];
        y = mc[i+1];
        z = mc[i+2];
        minx = std::min(minx,x);
        miny = std::min(miny,y);
        minz = std::min(minz,z);

        maxx = std::max(maxx,x);
        maxy = std::max(maxy,y);
        maxz = std::max(maxz,z);
    }
    median_x = (maxx+minx)/2;
    median_y = (maxy+miny)/2;
    median_z = (maxz+minz)/2;

    std::vector<double> result;
    result.push_back(median_x);
    result.push_back(median_y);
    result.push_back(median_z);

    return result;
}

void read_mesh_file(){
    // mesh_file = "coaxial_103mm_704MHz.mphtxt.h5";

    if (mesh_file=="")
    {
        std::cerr << "An error occurred: No mesh_file found" << std::endl;
        mesh_ok=false;
        std::exit(1);
    } 
    else 
    {
        mesh = std::make_shared<dolfin::Mesh>();
        dolfin::HDF5File hdf(mesh->mpi_comm(), mesh_file, file_mode);
        hdf.read(*mesh, "/mesh", false);
        mesh_subdomains = std::make_shared<dolfin::MeshFunction<std::size_t>>(mesh, mesh->topology().dim());
        mesh_boundaries = std::make_shared<dolfin::MeshFunction<std::size_t>>(mesh, mesh->topology().dim() - 1);

        hdf.read(*mesh_subdomains, "/subdomains");
        hdf.read(*mesh_boundaries, "/boundaries");
        
        N_cells = mesh->num_cells();

        bmesh = std::make_shared<dolfin::BoundaryMesh>(*mesh,"exterior");
        N_ext = bmesh->num_cells();
        mesh_center = mesh_center_point(bmesh);

        tree  = std::make_shared<dolfin::BoundingBoxTree>();
        btree = std::make_shared<dolfin::BoundingBoxTree>();

        tree->build(*mesh,3);
        btree->build(*bmesh,2);

        N_elems = N_ext;

        domains_map = mesh_subdomains->values();
        domains_map_size = mesh_subdomains->size();

        for (int i = 0; i < N_ext; i++)
        {
            lut_sense.push_back(false);
        }

        mesh_ok = true;
        
        // print_info();

        // std::cout << "Se ha leido del archivo xdmf!\n" << std::endl;
    }
}

bool read_field_data(bool build_lookup_table_in = true){
    build_lookup_table = build_lookup_table_in;
    if (data_file == "")
    {
        std::cerr << "7. An error occurred: No data_file found" << std::endl;
        mesh_ok=false;
        std::exit(1);
    }
    else
    {
    
        std::ifstream file(data_file);
        if (!file.is_open()) {
            std::cerr << "1. Failed to open the data file." << std::endl;
            std::exit(1);
        }
        
        std::string line;

        // Skip header lines
        for (int i = 0; i < 9; ++i) std::getline(file, line);

        std::vector<double> EX, EY, EZ;
        std::vector<double> X, Y, Z;
        
        // Read E B field and process
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double x, y, z;
            std::string ex, ey, ez;

            if (iss >> x >> y >> z >> ex >> ey >> ez) {

                // Real part of EX
                size_t index_real = ex.find_first_of("+-",1);
                if (index_real != std::string::npos){
                    EX.push_back(std::stod(ex.substr(0,index_real)));
                } else {
                    EX.push_back(0.0);
                }

                // Real part of EY
                index_real = ey.find_first_of("+-",1);
                if (index_real != std::string::npos){
                    EY.push_back(std::stod(ey.substr(0,index_real)));
                } else {
                    EY.push_back(0.0);
                }
                
                // Real part of EZ
                index_real = ez.find_first_of("+-",1);
                if (index_real != std::string::npos){
                    EZ.push_back(std::stod(ez.substr(0,index_real)));
                } else {
                    EZ.push_back(0.0);
                }

                X.push_back(x);
                Y.push_back(y);
                Z.push_back(z);
            } 
        }

        auto V = std::make_shared<MyLagrange::FunctionSpace>(mesh);

        auto Fex = std::make_shared<dolfin::Function>(V);
        auto Fey = std::make_shared<dolfin::Function>(V);
        auto Fez = std::make_shared<dolfin::Function>(V);
        
        // INICIO --- Leer EX, EY, EZ desde ficheros python */
        // std::ifstream file_ex_numpy("EX_numpy.txt");
        // double valuex;
        // EX.clear();
        // while (file_ex_numpy >> valuex) EX.push_back(valuex);
        //
        // std::ifstream file_ey_numpy("EY_numpy.txt");
        // double valuey;
        // EY.clear();
        // while (file_ey_numpy >> valuey) EY.push_back(valuey);
        //
        // std::ifstream file_ez_numpy("EZ_numpy.txt");
        // double valuez;
        // EZ.clear();
        // while (file_ez_numpy >> valuez) EZ.push_back(valuez);
        /* FIN --- Leer EX, EY, EZ desde ficheros python */
        
        Fex->vector()->set_local(EX);
        Fex->vector()->apply("insert");
        Fey->vector()->set_local(EY);
        Fey->vector()->apply("insert");
        Fez->vector()->set_local(EZ);
        Fez->vector()->apply("insert");

        auto campoEx_temp = std::make_shared<dolfin::Function>(V);
        auto campoEy_temp = std::make_shared<dolfin::Function>(V);
        auto campoEz_temp = std::make_shared<dolfin::Function>(V);

        MyLagrange::BilinearForm ax(V,V);
        MyLagrange::LinearForm Lx(V);
        Lx.g = Fex;
        
        dolfin::solve(ax == Lx, *campoEx_temp);
        campoEx_temp->set_allow_extrapolation(true);

        MyLagrange::BilinearForm ay(V,V);
        MyLagrange::LinearForm Ly(V);
        Ly.g = Fey;
        dolfin::solve(ay == Ly, *campoEy_temp);
        campoEy_temp->set_allow_extrapolation(true);

        MyLagrange::BilinearForm az(V,V);
        MyLagrange::LinearForm Lz(V);
        Lz.g = Fez;
        dolfin::solve(az == Lz, *campoEz_temp);
        campoEz_temp->set_allow_extrapolation(true);

        campoEx = campoEx_temp;
        campoEy = campoEy_temp;
        campoEz = campoEz_temp;

        // INICIO --- Escribir camposEx, Ey, Ez en ficheros para comparar */
        // dolfin::HDF5File campoEx_file(MPI_COMM_WORLD, "campoEx.h5", "w");
        // campoEx_file.write(*campoEx, "/campoEx");
        //
        // dolfin::HDF5File campoEy_file(MPI_COMM_WORLD, "campoEy.h5", "w");
        // campoEy_file.write(*campoEy, "/campoEy");
        //
        // dolfin::HDF5File campoEz_file(MPI_COMM_WORLD, "campoEz.h5", "w");
        // campoEz_file.write(*campoEz, "/campoEz");
        /* FIN --- Escribir camposEx, Ey, Ez en ficheros para comparar */

        // INICIO --- Comparar lectura EX, Ey, EZ con ficheros python */
        // std::ifstream file_ex_numpy("EX_numpy.txt");
        // std::vector<double> EX_numpy;
        // double value;
        // while (file_ex_numpy >> value) EX_numpy.push_back(value);
        // int number_differ = 0;
        // bool equal=true;
        // if (EX.size()!=EX_numpy.size())
        // {
        //     std::cout << "EX: Size is not equal!" << std::endl;
        //     std::exit(1);
        // }
        // for (int i=0;i<EX.size(); i++){
        //     if (EX[i] != EX_numpy[i]){
        //         // std::cout << "Not equal at i: " << i << std::endl;
        //         // std::cout << "EX      : " << EX[i] << std::endl;
        //         // std::cout << "EX_numpy: " << EX_numpy[i] << std::endl;
        //         number_differ++;
        //         equal = false;
        //     }
        // }
        // if (equal) {
        //     std::cout << "EX: Values and precision match!" << std::endl;
        // } else {
        //     std::cout << "EX: Values and precision differ! Cuantas son distintas: " << number_differ << std::endl;
        // }
        //
        // std::ifstream file_ey_numpy("EY_numpy.txt");
        // std::vector<double> EY_numpy;
        // while (file_ey_numpy >> value) EY_numpy.push_back(value);
        // number_differ = 0;
        // equal=true;
        // if (EY.size()!=EY_numpy.size())
        // {
        //     std::cout << "EY: Size is not equal!" << std::endl;
        //     std::exit(1);
        // }
        // for (int i=0;i<EY.size(); i++){
        //     if (EY[i] != EY_numpy[i]){
        //         // std::cout << "Not equal at i: " << i << std::endl;
        //         // std::cout << "EX      : " << EX[i] << std::endl;
        //         // std::cout << "EX_numpy: " << EX_numpy[i] << std::endl;
        //         number_differ++;
        //         equal = false;
        //     }
        // }
        // if (equal) {
        //     std::cout << "EY: Values and precision match!" << std::endl;
        // } else {
        //     std::cout << "EY: Values and precision differ! Cuantas son distintas: " << number_differ << std::endl;
        // }
        //
        // std::ifstream file_ez_numpy("EZ_numpy.txt");
        // std::vector<double> EZ_numpy;
        // while (file_ez_numpy >> value) EZ_numpy.push_back(value);
        // number_differ = 0;
        // equal=true;
        // if (EZ.size()!=EZ_numpy.size())
        // {
        //     std::cout << "EZ: Size is not equal!" << std::endl;
        //     std::exit(1);
        // }
        // for (int i=0;i<EZ.size(); i++){
        //     if (EZ[i] != EZ_numpy[i]){
        //         // std::cout << "Not equal at i: " << i << std::endl;
        //         // std::cout << "EX      : " << EX[i] << std::endl;
        //         // std::cout << "EX_numpy: " << EX_numpy[i] << std::endl;
        //         number_differ++;
        //         equal = false;
        //     }
        // }
        // if (equal) {
        //     std::cout << "EZ: Values and precision match!" << std::endl;
        // } else {
        //     std::cout << "EZ: Values and precision differ! Cuantas son distintas: " << number_differ << std::endl;
        // }
        /* FIN --- Comparar lectura EX, Ey, EZ con ficheros python */
        
        if (!build_lookup_table) return true;

        // Some field data and pre-processing to accelerate calculations
        for (size_t i = 0; i < N_ext; i++)
        {
            auto facet_i = dolfin::Face(*bmesh,i);
            auto mp = facet_i.midpoint();
            Eigen::VectorXd X0(3);
            X0 << mp[0],mp[1],mp[2];
            Eigen::Ref<const Eigen::VectorXd> ref_X0 = X0;
            
            Eigen::VectorXd res_ex(1);
            Eigen::Ref<Eigen::VectorXd> ref_res_ex = res_ex;
            campoEx->eval(ref_res_ex,ref_X0);

            Eigen::VectorXd res_ey(1);
            Eigen::Ref<Eigen::VectorXd> ref_res_ey = res_ey;
            campoEy->eval(ref_res_ey,ref_X0);

            Eigen::VectorXd res_ez(1);
            Eigen::Ref<Eigen::VectorXd> ref_res_ez = res_ez;
            campoEz->eval(ref_res_ez,ref_X0);

            Eigen::Vector3d EX0(ref_res_ex[0],ref_res_ey[0],ref_res_ez[0]);
            double E0 = EX0.norm();            
            // std::cout << "i: " << i << " E0: " << E0 << std::endl;
            // std::cout << "i: " << i << " X0: " << ref_X0 << std::endl; // Correcto
            // std::cout << "i: " << i << " EX0: " << EX0 << std::endl;
            lut_EX0.push_back(EX0);
            lut_E0.push_back(E0);
        }

        return true;
        
    }
}

void read_from_data_files(){
    read_mesh_file();
    read_field_data();

    verbose = true;
    if (verbose){
        std::cout << "Data file: " << data_file << std::endl;
        std::cout << "Mesh file: " << mesh_file << std::endl;
        std::cout << "Data read. Num surface elems: " << N_elems  << " Num volume elems: " << N_cells << std::endl;
    }
}

double closest_entity(double Xp, double Yp, double Zp){
    std::stringstream sstream;
    sstream << std::fixed << std::setprecision(4) << Xp << "_" << Yp << "_" << Zp;
    std::string strx = sstream.str();

    double D;  

    if (closest_entity_dictionary.find(strx) != closest_entity_dictionary.end()){
        D = closest_entity_dictionary[strx];
    } else {
        dolfin::Point point(Xp,Yp,Zp);
        auto D_pair = tree->compute_closest_entity(point);
        D = D_pair.second;
        closest_entity_dictionary[strx] = D;
    }

    return D;
}

double secondary_electron_yield(double ev)
{
    double S = 0;
    try 
    {
        S = 0.003*std::pow(ev,1.3)*std::exp(-0.003*ev)+0.2+0.5*std::exp(-0.01*ev);
        // std::cout << "Valor S=" << S << " con energy_eV=" << ev << std::endl;    
    } 
    catch (std::overflow_error)
    {
        std::cout << "Overflow in exponential math.exp. Energy=" << ev << std::endl;
    }
    return S;
}

int probability_of_emission(double sey)
{
    // No entiendo el por qué del '+1' así que lo vuelvo a poner al original por si acaso al final
    if (random_seed != -1) srand(random_seed+1); // To generate the same electrons regardless of the order in which they are simulated
    double x = (double) rand()/RAND_MAX;
    double y = 0, n_fact = 1;
    int n = 0;
    while (y < x)
    {
        double p = std::pow(sey,n)*std::exp(-sey);
        p = p/n_fact;
        y = y + p;
        n = n + 1;
        n_fact = n_fact * n;
    }
    if (random_seed != -1) srand(random_seed);
    return n-1;
}

std::tuple<int,std::vector<double>,double> total_secondary_electrons(double energy_eV)
{
    double sey = secondary_electron_yield(energy_eV);
    int n = probability_of_emission(sey);
    std::vector<double> energies;
    double bote = energy_eV;
    for (int i=0;i<n;i++)
    {
        double y = (double) rand()/RAND_MAX;
        double e0 = y*bote;
        energies.push_back(e0);
        bote = bote - e0;
    }

    return std::make_tuple(n,energies,sey);

}

std::vector<int> remove_by_boolean_condition (std::vector<int> test_face, double operation_in, double coordinate, double value, double tolerance){
    std::vector<int> new_lista;
    bool condition = false;
    // int operation = std::round(operation_in);
    int operation = static_cast<int>(operation_in);

    for (int s : test_face)
    {
        auto facet_i = dolfin::Face(*bmesh,s);
        auto mp = facet_i.midpoint();
        condition = false;
        auto x = mp[coordinate];
        switch (operation)
        {
            case 0:
                condition = std::abs(x-value)<tolerance;
                break;
            case -1:
                condition = x<value;
                break;
            case 1:
                condition = x>value;
                break;
            default:
                std::cerr << "Unknown operation, only -1, 0 or 1 accepted" << std::endl;
                std::exit(1);
                break;
        }
        if (!condition) new_lista.push_back(s);
    }
    return new_lista;
}

std::vector<std::vector<double>> get_initial_conditions_face(int face_i){
    std::vector<std::vector<double>> result;
    std::vector<double> X0(3), U0(3);

    std::vector<double> nodes;
    for (int i = 0;i<3;i++) nodes.push_back(bmesh->cells()[3*face_i+i]);
    
    std::vector<std::vector<double>> X;
    std::vector<double> temp_list;
    auto coords = bmesh->coordinates();
    for (int i=0;i<3;i++) 
    {
        for (int j=0;j<3;j++)
        {
            temp_list.push_back(coords[3*nodes[i]+j]);
        }
        X.push_back(temp_list);
        temp_list.clear();
    }

    auto facet_i = dolfin::Face(*bmesh,face_i);
    auto mp = facet_i.midpoint();
    // std::cout << "face_i: " << face_i << " mp is: " << mp[0] << "," << mp[1] << "," << mp[2] << std::endl;
    auto Nv = face_normal(X);
    // std::cout << " Nv is: " << Nv[0] << "," << Nv[1] << "," << Nv[2] << std::endl;

    int sense_factor_int = 1;
    if (!lut_sense[face_i])
    {
        auto Xc = mesh_center;
        auto dcm = std::sqrt((std::pow((mp[0]-Xc[0]),2))+(std::pow((mp[1]-Xc[1]),2))+( std::pow((mp[2]-Xc[2]),2)));

        double Xp = mp[0] + (dcm * 0.1 * Nv[0]);
        double Yp = mp[1] + (dcm * 0.1 * Nv[1]);
        double Zp = mp[2] + (dcm * 0.1 * Nv[2]);

        double dcm_d = closest_entity(Xp,Yp,Zp);
        if (dcm_d > 0.0) sense_factor_int = -1; // Punto fuera del mallado
        // Update look-up table
        if (sense_factor_int == -1) lut_sense[face_i] = false;
        else lut_sense[face_i] = true;
    }
    // std::cout << "Sense_factor_int: " << sense_factor_int << std::endl;
    
    for (int i=0;i<3;i++)
    {
        X0[i] = mp[i];
        U0[i] = (Nv[i]*sense_factor_int);
    }
    
    std::vector<double> EX0(lut_EX0[face_i].data(),lut_EX0[face_i].data() + lut_EX0[face_i].size());
    
    result.push_back(X0);
    result.push_back(U0);
    result.push_back(EX0);

    // std::cout << " X0 is: " << X0[0] << "," << X0[1] << "," << X0[2] << std::endl;
    // std::cout << " U0 is: " << U0[0] << "," << U0[1] << "," << U0[2] << std::endl;
    // std::cout << " EX0 is: " << EX0[0] << "," << EX0[1] << "," << EX0[2] << std::endl;

    return result;
}

std::tuple<unsigned int, double, double, std::vector<std::vector<double>>> track_1_e (double electron_energy = -1, double power = 1.0, double phase = 0.0, int face_i = -1, bool keep = false, bool show_in = false){
    /* Runs the tracking of 1 electron in the problem geometry.
    power: RF power [W] in the device

    phase: phase [rad] when eletron is emmited (field will be E=E0 cos (wt+phase)

    face_i: the surface facet element where the electron is emmited. If not
    especified it is chosen ramdomly.

    keep: [False] Wheather or not to keep the full trayectory. If
    show_in==True, this is automatically also True

    show_in: [False] Shows the mesh and the electron trayectory.

    Return values: [collision, energy_collision]
        collision: face index where electron ended (or None)
        energy_collision: energy [eV] of the electron when collision happens
    */
//    face_i = 1102;
//    log_exec = true;
   if (log_exec)
   {
        std::ofstream logfile(logfile_name);
        if (!logfile.is_open()) {std::cerr << "2. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
        logfile << "Call track_1_e starting from face= " << face_i << ", phase= " << phase << ", energy=" << electron_energy << std::endl;
        logfile.close();
   }

   double field_factor = std::sqrt(power);

   if (magnetic_field_on)
   {
        // Qué pasa aquí??
        // campoBx=self.campoBx
        // campoBy=self.campoBy
        // campoBz=self.campoBz
   }

    std::vector<std::vector<double>> trayectoria;
    std::vector<double> energia_electron;
    unsigned int collision = 0;
    double energy_collision;
    double phase_collision;

    if (electron_energy == -1) electron_energy = energy_0;

    double max_gamma = 0;
    if (show_in) keep = true;

    
    // Me salto el else del (if not starting_point) porque nunca se le pasa uno y son estructuras complicadas en C++ (un Eigen::vectorXd y otra cosa más)
    if (face_i == -1) face_i = rand() % N_ext;
    // std::cout << "Starting with face_i: " << face_i << std::endl;
    auto result = get_initial_conditions_face(face_i);
    std::vector<double> X0  = result[0];
    std::vector<double> U0  = result[1];
    std::vector<double> EX0 = result[2];

    // Initial velocity: computed from electron_energy
    std::vector<double> V0(3);
    double gamma = 1.0 + (electron_energy*electron_e_over_mc2);
    double beta  = std::sqrt((gamma*gamma)-1)/gamma;
    double v0    = beta * speed_light_vacuum;
    
    for (int i=0;i<3;i++) V0[i] = U0[i]*v0;

    double t = 0;
    double w = angular_frequency;
    double alpha = (w*t) + phase;
    std::vector<double> EX(3);
    for (int i=0;i<3;i++) EX[i] = (EX0[i]*field_factor*std::cos(alpha));

    // Tracking
    bool ended = false;
    std::vector<double> VX(V0);
    std::vector<double> X(X0);
    double t_max = N_cycles/RF_frequency;

    std::vector<double> P0(3);
    for (int i=0;i<3;i++) P0[i] = ((gamma/speed_light_vacuum)*V0[i]);
    // int count = 0;

    // std::cout << " VX is: " << VX[0] << "," << VX[1] << "," << VX[2] << std::endl;
    // std::cout << " V0 is: " << V0[0] << "," << V0[1] << "," << V0[2] << std::endl;
    // std::cout << " EX is: " << EX[0] << "," << EX[1] << "," << EX[2] << std::endl;
    // std::cout << " EX0 is: " << EX0[0] << "," << EX0[1] << "," << EX0[2] << std::endl;
    // std::cout << " X is: " << X[0] << "," << X[1] << "," << X[2] << std::endl;
    // std::cout << " X0 is: " << X0[0] << "," << X0[1] << "," << X0[2] << std::endl;
    // std::cout << " U0 is: " << U0[0] << "," << U0[1] << "," << U0[2] << std::endl;
    // std::cout << " delta_t is: " << delta_t << std::endl;
    // std::cout << " t_max is: " << t_max << std::endl;
    // std::cout << " P0 is: " << P0[0] << "," << P0[1] << "," << P0[2] << std::endl;
    // std::cout << " alpha is: " << alpha << std::endl;
    // std::cout << " gamma is: " << gamma << std::endl;
    // std::cout << " w is: " << w << std::endl;
    // std::cout << " power is: " << power << std::endl;
    // std::cout << " field_factor is: " << field_factor << std::endl;
    // std::cout << " t is: " << t << std::endl;

    while (!ended)
    {
        // count ++;
        EX0 = evaluate_campo_from_point(X[0],X[1],X[2]);
        // std::cout << "EX0: " << EX0[0] << "," << EX0[1] << "," << EX0[2] << std::endl; 
        // std::cout << "V0: " << V0[0] << "," << V0[1] << "," << V0[2] << std::endl; 
        alpha = (w*t) + phase;
        for (int i=0;i<3;i++) EX[i] = (EX0[i]*field_factor*std::cos(alpha));

        if (!magnetic_field_on)
        {
            // if (count < 10) std::cout << "Magnetic field is OFF" << std::endl;
            std::vector<double> P_minus(3), P(3);
            for (int i=0;i<3;i++) P_minus[i] = (P0[i]-(electron_e_over_2mc*EX[i]*delta_t));
            std::vector<double> P_plus(P_minus);
            for (int i=0;i<3;i++) P[i] = (P_plus[i]-(electron_e_over_2mc*EX[i]*delta_t));
            gamma = std::sqrt(1.0 + P[0]*P[0] + P[1]*P[1] + P[2]*P[2]);
            for (int i=0;i<3;i++) 
            {
                VX[i] = speed_light_vacuum*P[i]/gamma;
                P0[i] = P[i];
            }

        }
        else
        {
            // Aquí se usa campoBx/By/Bz pero no se inicializa nunca?? Usaré Ex/Ey/Ez por ahora
            std::vector<double> BX = evaluate_campo_from_point(X[0],X[1],X[2]);
            std::vector<double> epsilon(3), T(3), U_minus(3), U_plus(3), U_prime(3), UX(3);
            for (int i=0;i<3;i++)
            {
                U0[i] = VX[i]*gamma;
                epsilon.push_back(electron_e_over_2m*EX[i]*(-1.0));
                T.push_back((electron_e_over_2m*delta_t*BX[i]*(-1.0))/gamma);
                U_minus.push_back(U0[i]+epsilon[i]*delta_t);
            }

            U_prime[0] = U_minus[0] + (U_minus[1]*T[2] - U_minus[2]*T[1]);
            U_prime[1] = U_minus[1] + (U_minus[2]*T[0] - U_minus[0]*T[2]);
            U_prime[2] = U_minus[2] + (U_minus[0]*T[1] - U_minus[1]*T[0]);

            double s = 2.0/(1+T[0]*T[0]+T[1]*T[1]+T[2]*T[2]);

            U_plus[0] = U_minus[0] + s*(U_prime[1]*T[2] - U_prime[2]*T[1]);
            U_plus[1] = U_minus[1] + s*(U_prime[2]*T[0] - U_prime[0]*T[2]);
            U_plus[2] = U_minus[2] + s*(U_prime[0]*T[1] - U_prime[1]*T[0]);   

            for (int i=0;i<3;i++)
            {
                UX[i] = U_plus[i] + epsilon[i]*delta_t;
                VX[i] = UX[i]/gamma;
            }         
            beta = std::sqrt(VX[0]*VX[0] + VX[1]*VX[1] + VX[2]*VX[2])/speed_light_vacuum;
            gamma = 1.0/(std::sqrt(1-beta*beta));
        }

        max_gamma = std::max(gamma,max_gamma);

        for (int i=0;i<3;i++)
        {
            X[i] = X[i] + VX[i]*delta_t;
            V0[i] = VX[i];
        }

        if (!point_inside_mesh(X))
        {
            // std::cout << "Se ha salido del mesh!" << std::endl;
            // Out of the mesh, this is a collision with a boundary
            double energia_eV = (gamma-1)/electron_e_over_mc2;
            if (gamma <= 1.0)
            {
                std::cerr << "gamma= " << gamma << " eV= " << energia_eV << std::endl;
                std::cerr << VX[0] << " " << VX[1] << " " << VX[2] << " " << std::endl;
                std::exit(1);
            }
            dolfin::Point point_x(X[0],X[1],X[2]);
            auto collision_face_pair = btree->compute_closest_entity(point_x);
            while (collision_face_pair.first >= N_ext)
            {
                std::cout << "Fallo de btree, con X={" <<X[0]<<","<< X[1]<<","<<X[2]<<"} y collision= " << collision_face_pair.first << "," << collision_face_pair.second << std::endl;
                int xyz = rand() % 3;
                // double lower_bound = 0;
                // double upper_bound = 1;
                // double sr = lower_bound + (upper_bound - lower_bound) * (rand() % RAND_MAX) / RAND_MAX;
                double sr = (double) rand()/RAND_MAX;
                sr = sr*0.002 - 0.001;
                std::vector<double> Y = {0,0,0};
                Y[xyz] = sr;
                X[0] += Y[0];
                X[1] += Y[1];
                X[2] += Y[2];
                dolfin::Point point_x(X[0],X[1],X[2]);
                collision_face_pair = btree->compute_closest_entity(point_x);
            }
            collision = collision_face_pair.first;
            energy_collision = energia_eV;
            phase_collision = alpha;
            ended = true;
        }

        if (keep)
        {
            double energia_eV=(gamma-1)/electron_e_over_mc2;
            trayectoria.push_back(X);
            energia_electron.push_back(energia_eV);
        }

        t = t + delta_t;
        if (t > t_max) {ended = true; std::cout << "Se acabó el tiempo!" << std::endl;}

    } /* Termina el while not ended*/
    
    // std::cout << "Number of displacements count is: " << count << std::endl;
    
    if (show_in)
    {
        // Pendiente, por ahora nada
    }

    if (keep)
    {
        std::ofstream ftraj("generated_files/ultima_trayectoria.txt");
        if (!ftraj.is_open()) {std::cerr << "3. Error opening trayectory file: generated_files/ultima_trayectoria.txt" << std::endl; std::exit(1);}
        for (auto element : trayectoria) ftraj << element[0] << "\t" << element[1] << "\t" << element[2] << std::endl; 
        ftraj.close();

        std::ofstream fener("generated_files/ultima_energia.txt");
        if (!fener.is_open()) {std::cerr << "4. Error opening energy file: generated_files/ultima_energia.txt" << std::endl; std::exit(1);}
        for (auto element : energia_electron) fener << element << std::endl; 
        fener.close();

        auto now = std::time(nullptr);
        auto now_struct = *std::localtime(&now);
        std::ostringstream oss;
        oss << std::put_time(&now_struct, "%Y-%m-%dT%H:%M:%S");
        std::string now_str = oss.str();

        std::ofstream fnow("generated_files/" + now_str + ".txt");
        if (!fnow.is_open()) {std::cerr << "5. Error opening trayectory and energy file: generated_files/" << now_str << ".txt" << std::endl; std::exit(1);}
        for (int i=0;i<trayectoria.size();i++) fnow << trayectoria[i][0] << " " << trayectoria[i][1] << " " << trayectoria[i][2] << " " << energia_electron[i] << std::endl;
        fnow.close();
    }

    if (log_exec)
    {
        std::ofstream logfile(logfile_name);
        if (!logfile.is_open()) {std::cerr << "6. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
        logfile << "Completed, collision= " << collision << std::endl; 
        logfile.close();
    }

    if (collision == 0)
    {
        double energia_eV=(max_gamma-1)/electron_e_over_mc2;
        energy_collision = energia_eV;
    }

    return std::make_tuple(collision,energy_collision,phase_collision,trayectoria);
}

std::tuple<unsigned int, double, double, std::vector<std::vector<double>>> run_1_electron(double power, int face, double rf_phase, double energy_0_copy, bool keep = false)
{
    std::chrono::steady_clock::time_point start_1e = std::chrono::steady_clock::now();

    unsigned int collision;
    double energy_collision, phase_collision;
    std::vector<std::vector<double>> trayectoria;
    std::tie(collision, energy_collision, phase_collision, trayectoria) = track_1_e(energy_0_copy,power,rf_phase,face,keep);

    std::chrono::steady_clock::time_point end_1e = std::chrono::steady_clock::now();

    // Guardar tiempo 1 electron
    int segundos_1e     = std::chrono::duration_cast<std::chrono::seconds>(end_1e - start_1e).count();
    int milisegundos_1e = std::chrono::duration_cast<std::chrono::milliseconds>(end_1e - start_1e).count();

    std::ofstream logtime(logtime_name, std::ios::app);
    if (!logtime.is_open()) {std::cerr << "11. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
    if (segundos_1e != 0) logtime << "time 1e: " << segundos_1e << "." << milisegundos_1e << " sec" << std::endl;
    else logtime << "time 1e: " << milisegundos_1e << " ms" << std::endl;
    logtime.close();

    // std::cout << "Segundos: " << segundos_1e << std::endl;
    // std::cout << "Milisegundos: " << milisegundos_1e << std::endl;

    return std::make_tuple(collision,energy_collision,phase_collision,trayectoria);
}

std::tuple<int, int> run_n_electrons_parallel (double power, std::vector<int> pool_runs, std::vector<double> pool_phase, std::vector<double> pool_energies){

    bool ended = false;

    // Mostrar y guardar estado inicial
    std::cout << "Power=" << power << "W, initial #electrons: " << pool_phase.size() << std::endl;
    std::ofstream logtime(logtime_name, std::ios::app);
    if (!logtime.is_open()) {std::cerr << "9. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
    logtime << "Power=" << power << "W, initial #electrons: " << pool_phase.size() << std::endl;
    logtime.close();

    if (log_exec)
    {
        std::ofstream logfile(logfile_name, std::ios::app);
        if (!logfile.is_open()) {std::cerr << "10. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
        logfile << "Power=" << power << "W, initial #electrons: " << pool_phase.size() << std::endl;
        logfile.close();
    }

    if (show)
    {
        // Por ahora nada
    }

    // Se puede mejorar este apaño, memory wise (?)
    std::map<std::string, double> lut_results_face;
    std::map<std::string, double> lut_results_energy;
    std::map<std::string, double> lut_results_phase;
    std::map<std::string, std::vector<std::vector<double>>> lut_results_trayectoria;

    int n = 0, number_electrons = 0;

    while (!ended)
    {
        std::chrono::steady_clock::time_point start_run = std::chrono::steady_clock::now();

        std::vector<int> new_pool_runs;
        std::vector<double> new_pool_phases, new_pool_energies;

        if (pool_runs.empty())
        {
            ended = true;
            continue;
        }

        int mw = parallel;
        if (pool_runs.size() < mw) mw = pool_runs.size();

        if (mw == 1) // Serial, único modo por ahora
        {
            unsigned int face;
            int erun;
            double efase, energy_0_copy, energy, phase;
            std::vector<std::vector<double>> trayectoria;

            for (int i=0;i<pool_runs.size();i++)
            {
                erun = pool_runs[i]; efase = pool_phase[i]; energy_0_copy = pool_energies[i];

                std::stringstream sstream;
                sstream << erun;
                sstream << std::fixed << std::setprecision(4) << "_" << efase << "_" << energy_0_copy;
                std::string lut_index = sstream.str();

                if (lut_results_face.find(lut_index) != lut_results_face.end()){
                    face         = lut_results_face[lut_index];
                    energy       = lut_results_energy[lut_index];
                    phase        = lut_results_phase[lut_index];
                    trayectoria  = lut_results_trayectoria[lut_index];
                } else {
                    std::tie(face, energy, phase, trayectoria) = run_1_electron(power,erun,efase,energy_0_copy,show);
                    lut_results_face[lut_index]         = face;
                    lut_results_energy[lut_index]       = energy;
                    lut_results_phase[lut_index]        = phase;
                    lut_results_trayectoria[lut_index]  = trayectoria;
                }

                if (face != 0) // En track_1_e el valor por defecto es 0, si es distinto significa que sí se ha chocado 
                {
                    int n_e; std::vector<double> energies; double sey;
                    // std::cout << "Energy: " << energy << std::endl;
                    std::tie(n_e,energies,sey) = total_secondary_electrons(energy);
                    // std::cout << "n_e " << n_e << ", sey " << sey << ", energies [";
                    // for (auto element : energies) std::cout << element << " ";
                    // std::cout << "] " << std::endl;
                    if (log_exec)
                    {
                        std::ofstream logfile(logfile_name, std::ios::app);
                        if (!logfile.is_open()) {std::cerr << "12. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
                        logfile << "Completed run in face " << face << ", energy=" << energy << " eV, phase=" << phase << " rad" << std::endl;
                        logfile << "This produces " << sey << " sey and " << n_e << " new electrons" << std::endl << std::endl;
                        logfile.close();
                    }

                    for (int i=0;i<n_e;i++)
                    {
                        new_pool_runs.push_back(face);
                        new_pool_phases.push_back(phase);
                    }
                    for (auto element : energies) new_pool_energies.push_back(element);
                }               

            }
        } 
        else // Parallelo, sin implementar por el momento
        { 
            std::cerr << "Sin funcionalidad en paralelo aún" << std::endl;
            std::exit(1);
        }

        pool_runs = new_pool_runs;
        pool_phase = new_pool_phases;
        pool_energies = new_pool_energies;

        // Guardar tiempo de una 'run'
        std::chrono::steady_clock::time_point end_run = std::chrono::steady_clock::now();

        double macrosec_run = std::chrono::duration_cast<std::chrono::microseconds>(end_run - start_run).count();
        double minutos_run = macrosec_run/60000000;

        std::ofstream logtime(logtime_name, std::ios::app);
        if (!logtime.is_open()) {std::cerr << "13. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
        if (minutos_run >= 1) logtime << "Completed run " << n << ", time: " << minutos_run << " min, electrons alive: " << pool_runs.size() << std::endl;
        else logtime << "Completed run " << n << ", time: " << minutos_run*60 << " sec, electrons alive: " << pool_runs.size() << std::endl;
        logtime.close();

        // Mostrar tiempo de una 'run'
        std::cout << "Time of run: " << std::setprecision(2) << minutos_run << " min" << std::endl;
        std::cout << "Power=" << power << " W, run#: " << n << ", electrons alive:" << pool_runs.size() << std::endl;

        if (log_exec)
        {
            std::ofstream logfile(logfile_name, std::ios::app);
            if (!logfile.is_open()) {std::cerr << "14. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
            logfile << "Completed secondary run #: " << n << ", power=" << power << " W, electrons alive:" << pool_runs.size() << std::endl << std::endl;
            logfile.close();
        }

        if (show)
        {
            //Sin implementar por el momento
        }

        number_electrons = number_electrons + pool_runs.size();
        n = n+1;
        if (n>N_max_secondary_runs)
        {
            ended = true;
            std::cout << "Max number of secondary runs achieved at P=" << power << " W" << std::endl;
            if (log_exec)
            {
                std::ofstream logfile(logfile_name, std::ios::app);
                if (!logfile.is_open()) {std::cerr << "15. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
                logfile << "Max number of secondary runs achieved at P=" << power << " W" << std::endl;
                logfile.close();
            }
        }

        if (!ended) ended = pool_runs.empty();

    }

    if (show)
    {
        // Por ahora nada
    }

    return std::make_tuple(number_electrons,pool_runs.size());
}

int run(){
    // random_seed = -1; // AQUIIIIIIIIIIIIIIIIIIIIIii
    time_t start_total = std::time(nullptr);
    std::cout << "random_seed: " << random_seed << std::endl;
    if (random_seed == -1) srand (time(NULL));
    else srand(random_seed);

    tm start_total_struct = *std::localtime(&start_total);
    std::ostringstream oss;
    oss << std::put_time(&start_total_struct, "%Y%m%d_%H%M%S");

    std::string now_str = oss.str();
    logfile_name = "generated_files/log_mpc_py_" + now_str + ".txt";

    if (log_exec && (simulation_type != 2))
    {
        std::ofstream logfile(logfile_name);
        if (!logfile.is_open()) {std::cerr << "7. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
        logfile.close();
    }

    logtime_name = "generated_files/exec_time_" + now_str + ".txt";
    std::ofstream logtime(logtime_name);
    if (!logtime.is_open()) {std::cerr << "8. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
    logtime.close();

    // simulation_type = 2; // AQUIIIIIIIIIIIIIIIIIIIIIii
    
    if (simulation_type == 2)
    {
        std::vector<int> test_face;
        while (test_face.empty())
        {
            test_face.push_back(rand() % N_ext);
            // Remove by coordinate, boolean
            for (std::vector<double> ct : boundaries_excluded_boolean)
            {
                double operation  = ct[0];
                double coordinate = ct[1];
                double value      = ct[2];
                double tolerance  = ct[3];
                test_face = remove_by_boolean_condition (test_face, operation, coordinate, value, tolerance);
            }
        }
        double collision, energy_collision, phase_collision;
        std::vector<std::vector<double>> trayectoria;
        std::tie(collision, energy_collision, phase_collision, trayectoria) = track_1_e(-1.0,RF_power[0],0.0,-1,true);
        return 1;
    } 
    else if (simulation_type == 3) N_runs_per_power = 1;
    else if (simulation_type == 1) N_runs_per_power = electrons_seed;
    
    std::cout << "---Imprimiendo power---" << std::endl;
    for (auto element : RF_power) std::cout << element << " ";
    std::cout << std::endl;

    std::vector<int> total_electrons, final_electrons, power_partial;
    int number_of_electrons, electrons_last_cycle;
    for (const auto power : RF_power)
    {
        std::vector<int> pool_runs, new_pool_runs;
        std::vector<double> pool_energies, pool_phase;

        for (int i=0;i<N_runs_per_power;i++) new_pool_runs.push_back(rand() % N_elems); 

        std::cout << "Power: " << power << ", new_pool_runs: ";
        for (auto element : new_pool_runs) std::cout << element << " ";
        std::cout << std::endl;
        for (auto element : RF_power) std::cout << element << " ";
        std::cout << std::endl;

        for (std::vector<double> ct : boundaries_excluded_boolean)
        {
            double operation  = ct[0];
            double coordinate = ct[1];
            double value      = ct[2];
            double tolerance  = ct[3];
            new_pool_runs = remove_by_boolean_condition (new_pool_runs, operation, coordinate, value, tolerance);
        }

        for (auto element : new_pool_runs) pool_runs.push_back(element);
        for (auto _ : pool_runs){
            double random_double = (double) rand()/RAND_MAX;
            pool_phase.push_back(random_double*2.0*boost::math::constants::pi<double>());
            pool_energies.push_back(energy_0);
        } 

        std::tie(number_of_electrons, electrons_last_cycle) = run_n_electrons_parallel (power, pool_runs, pool_phase, pool_energies);
    
        total_electrons.push_back(number_of_electrons);
        final_electrons.push_back(electrons_last_cycle);
        power_partial.push_back(power);
    }

    std::ofstream calculo_multipacing("generated_files/calculo_multipacting.txt");
    if (!calculo_multipacing.is_open()) {std::cerr << "16. Error opening calculo_multipacting.txt file" << std::endl; std::exit(1);}
    for (int i=0;i<total_electrons.size();i++) calculo_multipacing << power_partial[i] << "\t" << total_electrons[i] << "\t" << final_electrons[i] << std::endl;
    calculo_multipacing.close();

    if (plot)
    {
        // Sin implementar aún
    }

    // Mostrar tiempo total
    time_t end_total = std::time(nullptr);
    tm end_total_struct = *std::localtime(&end_total);

    double minutes_total = (end_total_struct.tm_min - start_total_struct.tm_min);
    double seconds_total = (end_total_struct.tm_sec - start_total_struct.tm_sec);
    double hours_total = (end_total_struct.tm_hour - start_total_struct.tm_hour);
    
    minutes_total += (seconds_total/60) + (hours_total*60);
    std::cout << "Total time: " << minutes_total << " min" << std::endl;

    // Guardar tiempo total
    std::ofstream logtime2(logtime_name);
    if (!logtime2.is_open()) {std::cerr << "17. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
    if (minutes_total>=1) logtime2 << "Total execution time: " << std::setprecision(3) << minutes_total << " min" << std::endl;
    else logtime2 << "Total execution time: " << std::setprecision(3) << minutes_total*60 << " sec" << std::endl;
    logtime2.close();


    return 0;

}

int main(int argc, char* argv[]) {
    const float VERSION = 0.9;

    if (argc < 2)
    {
        std::cerr << "mpc_run version = " << VERSION << " - Run multipacting calculations (jlmunoz@essbilbao.org)" << std::endl;
        std::cerr << "See problem file (*.mpc) for description of parameters." << std::endl;
        std::cerr << "Syntax: python mpc_run.py problem.mpc" << std::endl;
        std::cerr << "(If you need an example .mpc file, `python mpc_run.py test` will generate test.mpc for you.)" << std::endl;
        std::exit(1);
    }

    std::ifstream mpc_file(argv[1]);
    // std::ifstream mpc_file("coaxial_704_01.mpc");

    if (!mpc_file)
    {
        std::cerr << "Error opening the file, ending execution" << std::endl;
        std::exit(1);
    }

    std::string line;
    std::pair<std::string,std::string> param_argument;

    while (std::getline(mpc_file, line))
    {
        param_argument = get_param_arg(line);
        if (param_argument.first != "")
        {
            if (param_argument.first == "boundaries_excluded_boolean") p[param_argument.first] = p[param_argument.first] + param_argument.second;
            else p[param_argument.first] = param_argument.second;
        }
    }
    
    mpc_file.close();

    set_parameters_dictionary();

    read_from_data_files();

    
    run();

    return 0;
}
