#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <dolfin.h>
#include <dolfin/io/HDF5File.h>
#include <mpi.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <boost/units/systems/si/codata/electromagnetic_constants.hpp>
#include <boost/units/systems/si/codata/electron_constants.hpp>
#include <boost/units/systems/si/codata/universal_constants.hpp>
#include <boost/math/constants/constants.hpp>
#include <typeinfo>
#include <omp.h>

#include "MyLagrange.h"
#include "electron_list.h"

// Class variables
bool log_exec = false;
bool plot = false;
bool verbose = false;
bool magnetic_field_on;
bool mesh_ok = false;
bool build_lookup_table;
bool show = false;

int electrons_seed = 100;
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

std::vector<double> coords_bmesh;
std::vector<unsigned int> cells_bmesh;

std::vector<std::vector<double>> lut_EX0;

std::size_t* domains_map = nullptr;

std::vector<bool> lut_sense;

std::map<std::string, double> closest_entity_dictionary;

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

unsigned long long int pasos_simulados=0;

/**
 * @brief Calcula el vector normal, normalizado, de un triángulo 3D.
 * 
 * A partir de tres puntos 3D, la función calcula el vector normal, normalizado, de la cara triangular
 * que forman los tres puntos. 
 *
 * @param X Un vector que contiene tres puntos 3D. Se espera que X tenga tamaño 3 y que cada vector interno tenga tamaño 3.
 *          El orden de los puntos sí altera el resultado final.
 * @return std::vector<double> Un vector normal 3D normalizado.
 */
std::vector<double> face_normal(std::vector<std::vector<double>> X){
    
    // Se extraen los puntos 3D
    std::vector<double> P1 = X[0];
    std::vector<double> P2 = X[1];
    std::vector<double> P3 = X[2];

    // Se calculan los vectores BmA (P1->P2) y CmA (P1->P3)
    std::vector<double> BmA = {(P2[0]-P1[0]),(P2[1]-P1[1]),(P2[2]-P1[2])};
    std::vector<double> CmA = {(P3[0]-P1[0]),(P3[1]-P1[1]),(P3[2]-P1[2])};

    // Se calcula el producto vectorial de BmA y CmA
    double N1 = BmA[1]*CmA[2] - BmA[2]*CmA[1];
    double N2 = BmA[2]*CmA[0] - BmA[0]*CmA[2];
    double N3 = BmA[0]*CmA[1] - BmA[1]*CmA[0];

    // Se calcula la magnitud del vector normal
    double mn = std::sqrt((N1*N1)+(N2*N2)+(N3*N3));

    // Se normaliza el vector normal
    std::vector<double> N = {(N1/mn),(N2/mn),(N3/mn)};

    return N;
}

/**
 * @brief Calcula la fuerza sufrida por el campo electromagnético en un punto 3D.
 * 
 * Esta función calcula la fuerza sufrida por el campo electromagnético 
 * en los 3 ejes a partir de un punto 3D dado.
 *
 * @param x Valor en el eje X del punto a evaluar
 * @param y Valor en el eje Y del punto a evaluar
 * @param z Valor en el eje Z del punto a evaluar
 * @return std::vector<double> que representa la fuerza sufrida en los ejes X,Y,Z
 */
std::vector<double> evaluate_campo_from_point(double x, double y, double z){

    // Se representa el punto usando un Eigen::vectorXd y su referencia
    Eigen::VectorXd point(3);
    point << x,y,z;
    Eigen::Ref<const Eigen::VectorXd> ref_point = point;

    // Se evalua la fuerza sufrida en el eje X según el punto
    Eigen::VectorXd res_ex(1);
    Eigen::Ref<Eigen::VectorXd> ref_res_ex = res_ex;
    campoEx->eval(ref_res_ex,ref_point);

    // Se evalua la fuerza sufrida en el eje Y según el punto
    Eigen::VectorXd res_ey(1);
    Eigen::Ref<Eigen::VectorXd> ref_res_ey = res_ey;
    campoEy->eval(ref_res_ey,ref_point);

    // Se evalua la fuerza sufrida en el eje Z según el punto
    Eigen::VectorXd res_ez(1);
    Eigen::Ref<Eigen::VectorXd> ref_res_ez = res_ez;
    campoEz->eval(ref_res_ez,ref_point);

    // Se agrupan los valores
    Eigen::Vector3d values(ref_res_ex[0],ref_res_ey[0],ref_res_ez[0]);

    // Se convierte a un std::vector<double> para más cómodo uso
    std::vector<double> result(values.data(),values.data() + values.size());


    return result;
}

/**
 * @brief Calcula la fuerza sufrida por el campo electromagnético en un punto 3D.
 * 
 * Esta función calcula la fuerza sufrida por el campo electromagnético 
 * en los 3 ejes a partir de un punto 3D dado.
 * Recibe 3 funciones por parámetro y hace el cálculo sobre esas funciones en lugar
 * de las funciones globales. Esta función se utiliza como alternativa thread-safe.
 *
 * @param x Valor en el eje X del punto a evaluar
 * @param y Valor en el eje Y del punto a evaluar
 * @param z Valor en el eje Z del punto a evaluar
 * @param campoEx_in Función a evaluar en el eje X
 * @param campoEy_in Función a evaluar en el eje Y
 * @param campoEz_in Función a evaluar en el eje Z
 * @return std::vector<double> que representa la fuerza sufrida en los ejes X,Y,Z
 */
std::vector<double> evaluate_campo_from_point_thread(double x, double y, double z, std::shared_ptr<dolfin::Function> campoEx_in, std::shared_ptr<dolfin::Function> campoEy_in, std::shared_ptr<dolfin::Function> campoEz_in){

    // Se representa el punto usando un Eigen::vectorXd y su referencia
    Eigen::VectorXd point(3);
    point << x,y,z;
    Eigen::Ref<const Eigen::VectorXd> ref_point = point;

    // Se evalua la fuerza sufrida en el eje X según el punto
    Eigen::VectorXd res_ex(1);
    Eigen::Ref<Eigen::VectorXd> ref_res_ex = res_ex;
    campoEx_in->eval(ref_res_ex,ref_point);

    // Se evalua la fuerza sufrida en el eje Y según el punto
    Eigen::VectorXd res_ey(1);
    Eigen::Ref<Eigen::VectorXd> ref_res_ey = res_ey;  
    campoEy_in->eval(ref_res_ey,ref_point);  

    // Se evalua la fuerza sufrida en el eje Z según el punto
    Eigen::VectorXd res_ez(1);
    Eigen::Ref<Eigen::VectorXd> ref_res_ez = res_ez;
    campoEz_in->eval(ref_res_ez,ref_point);

    // Se agrupan los valores
    Eigen::Vector3d values(ref_res_ex[0],ref_res_ey[0],ref_res_ez[0]);

    // Se convierte a un std::vector<double> para más cómodo uso
    std::vector<double> result(values.data(),values.data() + values.size());

    return result;
}

/**
 * @brief Evalua si un punto dado se encuentra dentro del mallado.
 * 
 * La función calcula si un punto 3D dado por parámetro se encuentra dentro del mallado.
 *
 * @param X Un vector que contiene un punto 3D. Se espera que X tenga tamaño 3
 * @return bool Indica si está dentro del mallado (true) o fuera (false)
 */
bool point_inside_mesh(std::vector<double> X){

    // Se convierte el punto 3D a un objeto reconocible por Dolfin
    dolfin::Point point(X[0], X[1], X[2]);

    // Se calcula la primera entidad con la que colisiona el punto
    unsigned int ent = tree->compute_first_entity_collision(point);

    // Si el identificador de entidad es mayor al número de entidades, significa que está fuera del mallado
    if (ent >= N_cells) return false;

    // Si hay dominios sólidos a considerar
    if (!solid_domains.empty())
    {
        // Se busca si la entidad con la que ha colisionado forma parte de un dominio sólido
        std::size_t domain_index = domains_map[ent];

        // Si se encuentra, el punto no está dentro del mallado
        if (std::find(solid_domains.begin(), solid_domains.end(),domain_index) != solid_domains.end()) return false;
    }

    // En otro caso, el punto está dentro del dominio
    return true;
}

/**
 * @brief Evalua si un punto dado se encuentra dentro del mallado.
 * 
 * La función calcula si un punto 3D dado por parámetro se encuentra dentro del mallado.
 * Recibe 1 BoundingBoxTree por parámetro y hace el cálculo sobre ese árbol en lugar
 * del global. Esta función se utiliza como alternativa thread-safe.
 *
 * @param X Un vector que contiene un punto 3D. Se espera que X tenga tamaño 3
 * @param tree_in BoundingBoxTree utilizado para comprobar las colisiones
 * @return bool Indica si está dentro del mallado (true) o fuera (false)
 */
bool point_inside_mesh_thread(std::vector<double> X, std::shared_ptr<dolfin::BoundingBoxTree> tree_in ){
    // Se convierte el punto 3D a un objeto reconocible por Dolfin
    dolfin::Point point(X[0], X[1], X[2]);

    // Se calcula la primera entidad con la que colisiona el punto
    unsigned int ent = tree_in->compute_first_entity_collision(point);

    // Si el identificador de entidad es mayor al número de entidades, significa que está fuera del mallado
    if (ent >= N_cells) return false;

    // Si hay dominios sólidos a considerar
    if (!solid_domains.empty())
    {
        // Se busca si la entidad con la que ha colisionado forma parte de un dominio sólido
        std::size_t domain_index = domains_map[ent];

        // Si se encuentra, el punto no está dentro del mallado
        if (std::find(solid_domains.begin(), solid_domains.end(),domain_index) != solid_domains.end()) return false;
    }

    // En otro caso, el punto está dentro del dominio
    return true;
}

/**
 * @brief Procesa un string para obtener el parámetro y su valor correspondiente.
 * 
 * Esta función procesa un string dado, y extrae el parámetro y su valor.
 * Se espera que el string dado sea de la forma: 'param=arg#comment'.
 *
 * @param line String con los datos
 * @return std::pair<std::string,std::string> equivalente a (param,arg) si el input es 'param=arg#comment'
 */
std::pair<std::string,std::string> get_param_arg(std::string line)
{
    // Si la línea está vacía, se devuelve un par vacío
    if (line.empty() || line.find_first_not_of(' ') == std::string::npos) return std::make_pair("","");

    // Si la línea es un comentario en sí, se devuelve un par vacío
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

/**
 * @brief Inicializa los dominios sólidos a partir de un string.
 * 
 * Esta función inicializa los dominios sólidos del mallado según lo indique el string de entrada
 *
 * @param ls std::string que contiene información sobre los dominios que deben considerarse sólidos.
 *           Se espera que su forma sea del estilo: '[1,3]'.
 */
void set_solid_domains_COMSOL(std::string ld){
    
    std::string aux;

    // Ee recorren los carácteres de ld, y se comprueba si son dígitos
    for (char c : ld) if (std::isdigit(c)) 
    {
        // En tal caso, se añade el número siguiente a la lista de dominios sólidos
        aux = c;
        solid_domains.push_back(std::stoi(aux)+1);
    }
}

/**
 * @brief Inicializa parámetros para la ejecución.
 * 
 * Esta función procesa e inicializa varios parámetros usados a lo largo de la ejecución.
 * Para ello utiliza un std::map<std::string,std::string> donde están almacenados los valores.
 *
 */
void set_parameters_dictionary(){

    // Parámetros de tipo std::string
    data_file = "data/" + p["data_file"];
    mesh_file = "data/" + p["mesh_file"];

    // Parámetros de tipo bool
    log_exec = p["log"] == "True";
    verbose = p["verbose"] == "True";

    // Parámetros de tipo int
    N_runs_per_power     = stoi(p["electrons_seed"]);
    N_max_secondary_runs = stoi(p["N_max_secondary_runs"]);
    random_seed          = stoi(p["random_seed"]);
    N_cycles             = stoi(p["N_cycles"]);
    simulation_type      = stoi(p["simulation_type"]);

    // Si la palabra 'range' se encuentra en el campo de RF_power, se deben buscar los valores de ese rango
    if (p["RF_power"].find("range") != std::string::npos){ 
        
        // Se procesa el string completo, para buscar el valor de inicio, de final y el incremento
        std::string ps = p["RF_power"];
        std::size_t index_start = ps.find('(');
        std::size_t index_end = ps.find(')');
        std::string ps_limpio = ps.substr(index_start+1, index_end - index_start - 1);
        
        std::size_t index_coma = ps_limpio.find(',');
        double x1 = stod(ps_limpio.substr(0,index_coma));

        std::string ps_limpio_2 = ps_limpio.substr(index_coma+1);
        index_coma = ps_limpio_2.find(',');
        double x2 = stod(ps_limpio_2.substr(0,index_coma));
        double x3 = stod(ps_limpio_2.substr(index_coma+1));
        
        // Se elimina el valor por defecto que tiene la lista antes de añadir todos los valores
        RF_power.clear();
        for (double i = x1; i < x2; i+=x3) RF_power.push_back(i);
        
    } 
    // Si no, solo debe añadirse un valor
    else 
    {
        double x1 = stod(p["RF_power"]);
        RF_power.clear();
        RF_power.push_back(x1);
    }

    // Parámetros de tipo double
    RF_frequency = stod(p["RF_frequency"]);
    angular_frequency = 2*M_PI*RF_frequency;
    delta_t = stod(p["delta_t"]);

    // Parámetros que puede haber o no
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

/**
 * @brief Calcula el punto central de un mallado dado.
 * 
 * A partir de un mallado, se recorren todos sus puntos y se calcula el punto medio total.
 *
 * @param mesh_in Objeto de tipo std::shared_ptr<dolfin::BoundaryMesh> que representa el mallado del cual 
 *                se quioere calcular el punto medio.
 * @return std::vector<double> Contiene 3 elementos que representan las coordenadas 3D del punto medio del mallado
 */
std::vector<double> mesh_center_point(std::shared_ptr<dolfin::BoundaryMesh> mesh_in){

    // Se obtienen todas las coordenadas 3D de todos los vértices del mallado
    std::vector<double> mc = mesh_in->coordinates();

    double minx=0, maxx=0;
    double miny=0, maxy=0;
    double minz=0, maxz=0;

    double x, y, z, median_x, median_y, median_z;

    // Se recorren todas las coordenadas
    for (int i = 0;i<mc.size();i+=3){
        // Se obtienen los puntos en los ejes
        x = mc[i];
        y = mc[i+1];
        z = mc[i+2];

        // Se actualizan los valores mínimos y máximos
        minx = std::min(minx,x);
        miny = std::min(miny,y);
        minz = std::min(minz,z);

        maxx = std::max(maxx,x);
        maxy = std::max(maxy,y);
        maxz = std::max(maxz,z);
    }

    // Se calcula el punto medio y se prepara el resultado
    std::vector<double> result = {(maxx+minx)/2,(maxy+miny)/2,(maxz+minz)/2};

    return result;
}

/**
 * @brief Lee el mallado desde fichero.
 * 
 * Esta función lee el mallado desde un archivo en formato HDF5. Calcula otros mallados secundarios a partir del 
 * mallado leído, así como otras estructuras de datos. Se espera que el fichero del mallado se haya especificado con anterioridad.
 *
 */
void read_mesh_file(){

    // Comprueba que ya hay un fichero guardado
    if (mesh_file=="")
    {
        std::cerr << "An error occurred: No mesh_file found" << std::endl;
        mesh_ok=false;
        std::exit(1);
    } 
    else 
    {
        // Se crea y se lee el objeto del mallado
        mesh = std::make_shared<dolfin::Mesh>();
        dolfin::HDF5File hdf(mesh->mpi_comm(), mesh_file, file_mode);
        hdf.read(*mesh, "/mesh", false);

        // Se crean y se leen las funciones de subdominio y borde
        mesh_subdomains = std::make_shared<dolfin::MeshFunction<std::size_t>>(mesh, mesh->topology().dim());
        mesh_boundaries = std::make_shared<dolfin::MeshFunction<std::size_t>>(mesh, mesh->topology().dim() - 1);

        hdf.read(*mesh_subdomains, "/subdomains");
        hdf.read(*mesh_boundaries, "/boundaries");
        
        // Se guarda el número de celdas en el mallado
        N_cells = mesh->num_cells();

        // Se crea el mallado que representa el exterior y se guarda el número de celdas y su centro
        bmesh = std::make_shared<dolfin::BoundaryMesh>(*mesh,"exterior");
        N_ext = bmesh->num_cells();
        mesh_center = mesh_center_point(bmesh);

        // Preparar lecturas para ahorrar tiempo
        coords_bmesh = bmesh->coordinates();
        cells_bmesh = bmesh->cells();

        // Se inicializan estructuras usadas para calcular las colisiones 
        tree  = std::make_shared<dolfin::BoundingBoxTree>();
        btree = std::make_shared<dolfin::BoundingBoxTree>();

        tree->build(*mesh,3);
        btree->build(*bmesh,2);

        N_elems = N_ext;

        domains_map = mesh_subdomains->values();

        // Se inicializa un sentido para cada una de las caras del mallado exterior
        for (int i = 0; i < N_ext; i++)
        {
            lut_sense.push_back(false);
        }

        mesh_ok = true;
    }
}

/**
 * @brief Lee de fichero información del campo electromagnético y crea las funciones correspondientes.
 * 
 * Lee información de un fichero que relaciona la fuerza electromagnética de cada eje en varios puntos del mallado.
 * Se espera que el fichero de datos se haya especificado con anterioridad.
 *
 * @param build_lookup_table_in indica si se debe acelerar cálculo, guardando la fuerza que sufre cada punto del mallado en una lista
 * @return bool que indica que la función ha terminado, true siempre
 */
bool read_field_data(bool build_lookup_table_in = true){
    build_lookup_table = build_lookup_table_in;

    // Se comprueba que hay aun fichero de datos
    if (data_file == "")
    {
        std::cerr << "7. An error occurred: No data_file found" << std::endl;
        mesh_ok=false;
        std::exit(1);
    }
    else
    {
        // Se comprueba que se ha podido abrir el fichero
        std::ifstream file(data_file);
        if (!file.is_open()) {
            std::cerr << "1. Failed to open the data file." << std::endl;
            std::exit(1);
        }
        
        std::string line;

        // Se saltan las primeras 9 líneas del fichero con información para el programador
        for (int i = 0; i < 9; ++i) std::getline(file, line);

        std::vector<double> EX, EY, EZ;
        std::vector<double> X, Y, Z;
        
        // Se lee cada línea, y se obtiene el punto 3D (X,Y,Z) y el valor de la fuerza electromagnética en los 3 ejes correspondientes 
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double x, y, z;
            std::string ex, ey, ez;

            if (iss >> x >> y >> z >> ex >> ey >> ez) {

                // Se extrae la parte real de EX (fuerza electromagnética del eje X)
                size_t index_real = ex.find_first_of("+-",1);
                if (index_real != std::string::npos){
                    EX.push_back(std::stod(ex.substr(0,index_real)));
                } else {
                    EX.push_back(0.0);
                }

                // Se extrae la parte real de EY (fuerza electromagnética del eje Y)
                index_real = ey.find_first_of("+-",1);
                if (index_real != std::string::npos){
                    EY.push_back(std::stod(ey.substr(0,index_real)));
                } else {
                    EY.push_back(0.0);
                }
                
                // Se extrae la parte real de EZ (fuerza electromagnética del eje Z)
                index_real = ez.find_first_of("+-",1);
                if (index_real != std::string::npos){
                    EZ.push_back(std::stod(ez.substr(0,index_real)));
                } else {
                    EZ.push_back(0.0);
                }

                // Se guardan los puntos 3D en distintas listas
                X.push_back(x);
                Y.push_back(y);
                Z.push_back(z);
            } 
        }

        // Se crea el espacio de funciones
        auto V = std::make_shared<MyLagrange::FunctionSpace>(mesh);

        // Se crean las funciones de prueba
        auto Fex = std::make_shared<dolfin::Function>(V);
        auto Fey = std::make_shared<dolfin::Function>(V);
        auto Fez = std::make_shared<dolfin::Function>(V);
        
        // Se asignan los puntos leídos desde el fichero a las funciones de prueba
        Fex->vector()->set_local(EX);
        Fex->vector()->apply("insert");
        Fey->vector()->set_local(EY);
        Fey->vector()->apply("insert");
        Fez->vector()->set_local(EZ);
        Fez->vector()->apply("insert");

        // Se crean las funciones donde se guardarán los resultados
        auto campoEx_temp = std::make_shared<dolfin::Function>(V);
        auto campoEy_temp = std::make_shared<dolfin::Function>(V);
        auto campoEz_temp = std::make_shared<dolfin::Function>(V);

        // Se crea el sistema bilineal y linear, y se asigna la función de prueba como coeficiente del sistema lineal
        MyLagrange::BilinearForm ax(V,V);
        MyLagrange::LinearForm Lx(V);
        Lx.g = Fex;
        
        // Se resuelve la igualdad y se obtiene la función resultado
        dolfin::solve(ax == Lx, *campoEx_temp);
        campoEx_temp->set_allow_extrapolation(true);

        // Se repite el proceso para cada eje
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

        // Se guardan las funciones para posterior uso
        campoEx = campoEx_temp;
        campoEy = campoEy_temp;
        campoEz = campoEz_temp;
        
        // Si no hay que adelantar cálculos, se termina la función
        if (!build_lookup_table) return true;

        // En caso contrario, se recorren todos los puntos del mallado exterior y se calcula y guarda la fuerza sufrida por el campo electromagnético
        for (size_t i = 0; i < N_ext; i++)
        {
            // Se obtiene la cara del mallado exterior
            dolfin::Face facet_i = dolfin::Face(*bmesh,i);

            // Se obtiene su punto medio
            dolfin::Point mp = facet_i.midpoint();

            // Se calcula la fuerza sufrida
            std::vector<double> EX0 = evaluate_campo_from_point(mp[0],mp[1],mp[2]);

            // Se calcula la magnitud del vector de fuerza 
            double E0 = std::sqrt(EX0[0] * EX0[0] + EX0[1] * EX0[1] + EX0[2] * EX0[2]);           
            
            // Se guardan los resultados para posterior uso
            lut_EX0.push_back(EX0);
            lut_E0.push_back(E0);
        }

        return true;
        
    }
}

/**
 * @brief Lee el fichero de datos y el fichero del mallado.
 * 
 * Esta función llama a las funciones correspondientes para leer los ficheros de datos .mpc y .txt, así como el fichero del mallado .h5
 * Si está el modo verbose, se imprime un mensage en consola diciendo que se han leído los ficheros y cuántos elementos hay en el exterior y en total del mallado.
 */
void read_from_data_files(){
    read_mesh_file();
    read_field_data();

    if (verbose){
        std::cout << "Data file: " << data_file << std::endl;
        std::cout << "Mesh file: " << mesh_file << std::endl;
        std::cout << "Data read. Num surface elems: " << N_elems  << " Num volume elems: " << N_cells << std::endl;
    }
}

/**
 * @brief Devuelve la distancia a la entidad más cercana al punto dado.
 * 
 * Esta función representa cada punto en un string y guarda en un mapa la distancia a la entidad más cercana a cada punto. 
 * Si el punto dado ya se ha tratado con anterioridad, se devuelve la distancia guardada asociada con tal punto. 
 * Si no, calcula la distancia y guarda el resultado antes de devolverlo.
 *
 * @param Xp Valor del punto en el eje X
 * @param Yp Valor del punto en el eje Y
 * @param Zp Valor del punto en el eje Z
 * @return double Distancia a la entidad más cercana al punto dado
 */
double closest_entity(double Xp, double Yp, double Zp){

    // Se representa el punto en forma de string
    std::stringstream sstream;
    sstream << std::fixed << std::setprecision(4) << Xp << "_" << Yp << "_" << Zp;
    std::string strx = sstream.str();

    double D;  

    // Si el punto ya se encontraba en el mapa, se devuelve su valor asociado
    if (closest_entity_dictionary.find(strx) != closest_entity_dictionary.end()){
        D = closest_entity_dictionary[strx];

    // Si no se encontraba en el mapa
    } else {
        // Se representa el punto en un objeto reconocible por Dolfin
        dolfin::Point point(Xp,Yp,Zp);

        // Se obtiene la entidad más cercana y su distancia
        std::pair<unsigned int, double> D_pair = tree->compute_closest_entity(point);
        D = D_pair.second;
        
        // Se guarda la distancia asociada al punto 
        closest_entity_dictionary[strx] = D;
    }

    return D;
}

/**
 * @brief Devuelve la distancia a la entidad más cercana al punto dado.
 * 
 * Esta función representa cada punto en un string y guarda en un mapa la distancia a la entidad más cercana a cada punto. 
 * Si el punto dado ya se ha tratado con anterioridad, se devuelve la distancia guardada asociada con tal punto. 
 * Si no, calcula la distancia y guarda el resultado antes de devolverlo.
 * Recibe 1 BoundingBoxTree por parámetro y hace el cálculo sobre ese árbol en lugar
 * del global. Esta función se utiliza como alternativa thread-safe.
 *
 * @param Xp Valor del punto en el eje X
 * @param Yp Valor del punto en el eje Y
 * @param Zp Valor del punto en el eje Z
 * @param tree_in BoundingBoxTree utilizado para comprobar la distancia a la entidad más próxima
 * @return double Distancia a la entidad más cercana al punto dado
 */
double closest_entity_thread(double Xp, double Yp, double Zp, std::shared_ptr<dolfin::BoundingBoxTree> tree_in){
    
    // Se representa el punto en forma de string
    std::stringstream sstream;
    sstream << std::fixed << std::setprecision(4) << Xp << "_" << Yp << "_" << Zp;
    std::string strx = sstream.str();

    double D;  

    // Si el punto ya se encontraba en el mapa, se devuelve su valor asociado
    if (closest_entity_dictionary.find(strx) != closest_entity_dictionary.end()){
        D = closest_entity_dictionary[strx];
    
    // Si no se encontraba en el mapa
    } else {
        // Se representa el punto en un objeto reconocible por Dolfin
        dolfin::Point point(Xp,Yp,Zp);

        // Se obtiene la entidad más cercana y su distancia
        std::pair<unsigned int, double> D_pair = tree_in->compute_closest_entity(point);
        D = D_pair.second;
        
        // Se guarda la distancia asociada al punto 
        #pragma omp critical (dictionary)
        {
            closest_entity_dictionary[strx] = D;
        }
    }

    return D;
}

/**
 * @brief Calcula el valor de producción de electrones secundarios.
 * 
 * Dada una energía, calcula el valor de producción de electrones secundarios siguiendo una fórmula concreta. 
 *
 * @param ev Energía con la que colisiona un electrón
 * @return double que representa el valor de producción de electrones secundarios. Devuelve 0 si el valor calculado es demasiado grande
 */
double secondary_electron_yield(double ev)
{
    double S = 0;
    try 
    {
        S = 0.003*std::pow(ev,1.3)*std::exp(-0.003*ev)+0.2+0.5*std::exp(-0.01*ev);
    } 
    catch (std::overflow_error)
    {
        std::cerr << "Overflow in exponential math.exp. Energy=" << ev << std::endl;
    }
    return S;
}

/**
 * @brief Calcula cuántos electrones se desprenden de una colisión según el valor de producción.
 * 
 * Esta función utiliza una aproximación de la distribución Poisson para determinar cuántos electrones
 * se desprenden en una colision, a partir del valor de producción de electrones dado por parámetro.
 * Para generar 'n' electrones hay una probabilidad 'p'. Entonces, para cada probabilidad 'p' hay un número de electrones
 * 'n' asociada. Esta función calcula la probabilidad de generar 0 electrones, y después la de generar 1 electrón, y así sucesivamente
 * acumulando la probabilidad total, hasta alcanzar una probabilidad acumulada arbitraria.
 *
 * @param sey Valor de producción de electrones
 * @return int Indica cuántos electrones se desprenden
 */
int probability_of_emission(double sey)
{
    // Se hace esta operación para generar la misma cantidad de electrones independientemente de la llamada
    if (random_seed != -1) srand(random_seed+1);

    // Se genera un número aleatorio entre 0 y 1 que sirve como barrera para la probabilidad acumulada
    double x = (double) rand()/RAND_MAX;
    double y = 0, n_fact = 1;
    int n = 0;

    // Mientras la probabilidad acumulada sea menor al número aleatorio
    while (y < x)
    {
        // Se calcula la probabilidad de que se generen 'n' electrones utilizando Poisson
        double p = std::pow(sey,n)*std::exp(-sey);
        p = p/n_fact;

        // Se acumula la probabilidad
        y = y + p;
        n = n + 1;
        n_fact = n_fact * n;
    }

    // Se reestablece la semilla anterior
    if (random_seed != -1) srand(random_seed);

    return n-1;
}

/**
 * @brief Calcula y devuelve los electrones generados, las energías que tienen y con qué valor de producción, a partir de la energía de una colisión.
 * 
 * Esta función calcula qué probabilidad de producción de electrones secundarios tiene la energía dada, cuántos electrones se generan (aleatoriamente) según ese valor,
 * y distribuye la energía de la colisión entre los electrones nuevos. Esta función es no determinista.
 *
 * @param energy_eV Energía de la colisión
 * @return std::tuple<int,std::vector<double>,double> que contiene cuántos electrones, sus energías y el valor de producción
 */
std::tuple<int,std::vector<double>,double> total_secondary_electrons(double energy_eV)
{
    // Se calcula el valor de producción según la energía de colisión
    double sey = secondary_electron_yield(energy_eV);

    // Se calcula cuántos electrones se generan, no determinista
    int n = probability_of_emission(sey);

    std::vector<double> energies;
    double bote = energy_eV;

    // A cada electrón se le asocia una energía
    for (int i=0;i<n;i++)
    {
        // La energía del electrón corresponde a un porcentaje aleatorio de la energía de la colisión
        double y = (double) rand()/RAND_MAX;
        double e0 = y*bote;
        energies.push_back(e0);

        // Se resta la energía asociada al electrón de la energía de la colisión, para conservar la energía total
        bote = bote - e0;
    }

    return std::make_tuple(n,energies,sey);
}

/**
 * @brief A partir de una lista y unas condiciones, elimina de la lista todos los valores que no cumplen las condiciones.
 * 
 * Esta función recibe una lista con identificadores de ciertas caras del mallado extrerno. 
 * Para cada cara, se busca su punto medio y se obtiene una de sus coordenadas a poner a prueba. 
 * Según el tipo de operación deseado, se pone a prueba esa coordenada, y si no pasa la prueba es eliminada de la lista. 
 * La función devuelve una lista con los identificadores de las caras que han pasado la prueba.
 *
 * @param test_face Lista de integers que representan las caras del mallado externo que se van a probar
 * @param operation_in double que representa el tipo de operación que se va a probar en cada cara
 * @param coordinate double que representa qué coordenada de los puntos medios de las caras se va a poner a prueba
 * @param value double que representa un valor que representa cosas distintas según el tipo de operación
 * @param tolerance double que representa qué tolerancia hay si la operación corresponde a la distancia del punto al valor
 * @return std::vector<int> que contiene los identificadores de las caras que han pasado la prueba
 */
std::vector<int> remove_by_boolean_condition (std::vector<int> test_face, double operation_in, double coordinate, double value, double tolerance){
    
    // Nueva lista que representa los elementos que no se han eliminado de la antigua lista
    std::vector<int> new_lista;

    // bool que indica si se ha cumplido la consición dada
    bool condition = false;

    // int que determina qué tipo de operación representa la condición
    int operation = static_cast<int>(operation_in);

    // Para cada elemento de la lista
    for (int s : test_face)
    {
        // Se obtiene la cara del mallado correspondiente
        dolfin::Face facet_i = dolfin::Face(*bmesh,s);

        // Se calcula su punto medio y se mira una de sus coordenadas (según parámetro)
        dolfin::Point mp = facet_i.midpoint();
        condition = false;
        double x = mp[coordinate];

        // Según el tipo de operación
        switch (operation)
        {
            // Si la distancia de la coordenada al valor de entrada es menor que la tolerancia
            case 0:
                condition = std::abs(x-value)<tolerance;
                break;

            // Si la coordenada es menor al valor de entrada
            case -1:
                condition = x<value;
                break;

            // Si la coordenada es mayor al valor de entrada
            case 1:
                condition = x>value;
                break;

            default:
                std::cerr << "Unknown operation, only -1, 0 or 1 accepted" << std::endl;
                std::exit(1);
                break;
        }

        // Si se ha cumplido la condición, se añade a la nueva lista
        if (!condition) new_lista.push_back(s);
    }

    return new_lista;
}

/**
 * @brief Obtiene condiciones iniciales de una cara según su identificador dado.
 * 
 * Esta función parte de un identificador de una cara de las celdas del mallado externo. Obtiene su punto medio, el vector normal de esa cara 
 * y la magnitud del vector normal. 
 *
 * @param face_i int que representa el identificador de la cara, para extraer sus condiciones, se espera que sea una cara válida
 * @return std::vector<std::vector<double>> que representa las coordenadas 3D del punto medio de la cara, el sentido del vector normal y la magnitud
 */
std::vector<std::vector<double>> get_initial_conditions_face(int face_i){

    // Se prepara el resultado, que consiste en las coordenadas 3D del punto medio de la cara, el sentido del vector normal y la magnitud
    std::vector<std::vector<double>> result;
    std::vector<double> X0(3), U0(3);

    // Se obtienen los identificadores de los vértices que forman la cara dada
    std::vector<double> nodes = {
        static_cast<double>(cells_bmesh[3*face_i+0]),
        static_cast<double>(cells_bmesh[3*face_i+1]),
        static_cast<double>(cells_bmesh[3*face_i+2])
    };

    // Se guardan las coordenadas 3D de cada uno de los vértices
    std::vector<std::vector<double>> X;
    std::vector<double> temp_list;
    for (int i=0;i<3;i++) 
    {
        for (int j=0;j<3;j++)
        {
            temp_list.push_back(coords_bmesh[3*nodes[i]+j]);
        }
        X.push_back(temp_list);
        temp_list.clear();
    }

    // Se obtiene el vector normal normalizado de la cara
    dolfin::Face facet_i = dolfin::Face(*bmesh,face_i);
    dolfin::Point mp = facet_i.midpoint();
    std::vector<double> Nv = face_normal(X);

    // Si el sentido de la cara es hacia fuera (no entiendo bien esto)
    int sense_factor_int = 1;
    if (!lut_sense[face_i])
    {
        // Se calcula la distancia desde el centro del mallado exterior a la cara
        std::vector<double> Xc = mesh_center;
        double dcm = std::sqrt((std::pow((mp[0]-Xc[0]),2))+(std::pow((mp[1]-Xc[1]),2))+( std::pow((mp[2]-Xc[2]),2)));

        // Se calcula un punto nuevo usando la distancia y el vector normal
        double Xp = mp[0] + (dcm * 0.1 * Nv[0]);
        double Yp = mp[1] + (dcm * 0.1 * Nv[1]);
        double Zp = mp[2] + (dcm * 0.1 * Nv[2]);

        // Se calcula la distancia de ese nuevo punto a la entidad más cercana
        double dcm_d = closest_entity(Xp,Yp,Zp);

        // Si la distancia es distinta de 0, significa que el punto se encuentra fuera del mallado
        if (dcm_d > 0.0) sense_factor_int = -1; 

        // Se actualiza la tabla que contiene el sentido de la cara
        if (sense_factor_int == -1) lut_sense[face_i] = false;
        else lut_sense[face_i] = true;
    }

    // Se prepara el punto 3D del centro de la cara
    X0[0] = mp[0];
    X0[1] = mp[1];
    X0[2] = mp[2];

    // Se prepara el sentido del vector normal
    U0[1] = (Nv[1]*sense_factor_int);
    U0[0] = (Nv[0]*sense_factor_int);
    U0[2] = (Nv[2]*sense_factor_int);
    
    // Se recoge la magnitud del vector normal
    std::vector<double> EX0(lut_EX0[face_i]);
    
    // Se prepara el resultado
    result.push_back(X0);
    result.push_back(U0);
    result.push_back(EX0);

    return result;
}

/**
 * @brief Obtiene condiciones iniciales de una cara según su identificador dado.
 * 
 * Esta función parte de un identificador de una cara de las celdas del mallado externo. Obtiene su punto medio, el vector normal de esa cara 
 * y la magnitud del vector normal. Recibe 1 BoundaryMesh del cual pertenece la cara a trabajar. También recibe 1 BoundingBoxTree por parámetro 
 * y hace el cálculo sobre ese árbol en lugar del global. Esta función se utiliza como alternativa thread-safe.
 *
 * @param face_i int que representa el identificador de la cara, para extraer sus condiciones, se espera que sea una cara válida
 * @param bmesh_in Mallado del cual se obtiene la cara a comprobar
 * @param tree_in BoundingBoxTree utilizado para comprobar si un punto está dentro del mallado
 * @return std::vector<std::vector<double>> que representa las coordenadas 3D del punto medio de la cara, el sentido del vector normal y la magnitud
 */
std::vector<std::vector<double>> get_initial_conditions_face_thread(int face_i, std::shared_ptr<dolfin::BoundaryMesh> bmesh_in,std::shared_ptr<dolfin::BoundingBoxTree> tree_in){
    
    // Se prepara el resultado, que consiste en las coordenadas 3D del punto medio de la cara, el sentido del vector normal y la magnitud
    std::vector<std::vector<double>> result;
    std::vector<double> X0(3), U0(3);

    // Se obtienen los identificadores de los vértices que forman la cara dada
    std::vector<double> nodes = {
        static_cast<double>(cells_bmesh[3*face_i+0]),
        static_cast<double>(cells_bmesh[3*face_i+1]),
        static_cast<double>(cells_bmesh[3*face_i+2])
    };
    
    // Se guardan las coordenadas 3D de cada uno de los vértices
    std::vector<std::vector<double>> X;
    std::vector<double> temp_list;
    for (int i=0;i<3;i++) 
    {
        for (int j=0;j<3;j++)
        {
            temp_list.push_back(coords_bmesh[3*nodes[i]+j]);
        }
        X.push_back(temp_list);
        temp_list.clear();
    }

    // Se obtiene el vector normal normalizado de la cara
    dolfin::Face facet_i = dolfin::Face(*bmesh_in,face_i);
    dolfin::Point mp = facet_i.midpoint();
    std::vector<double> Nv = face_normal(X);

    // Si el sentido de la cara es hacia fuera (no entiendo bien esto)
    int sense_factor_int = 1;
    if (!lut_sense[face_i])
    {
        // Se calcula la distancia desde el centro del mallado exterior a la cara
        std::vector<double> Xc = mesh_center;
        double dcm = std::sqrt((std::pow((mp[0]-Xc[0]),2))+(std::pow((mp[1]-Xc[1]),2))+( std::pow((mp[2]-Xc[2]),2)));

        // Se calcula un punto nuevo usando la distancia y el vector normal
        double Xp = mp[0] + (dcm * 0.1 * Nv[0]);
        double Yp = mp[1] + (dcm * 0.1 * Nv[1]);
        double Zp = mp[2] + (dcm * 0.1 * Nv[2]);

        // Se calcula la distancia de ese nuevo punto a la entidad más cercana
        double dcm_d = closest_entity_thread(Xp,Yp,Zp,tree_in);

        // Si la distancia es distinta de 0, significa que el punto se encuentra fuera del mallado
        if (dcm_d > 0.0) sense_factor_int = -1; 

        // Se actualiza la tabla que contiene el sentido de la cara
        if (sense_factor_int == -1) lut_sense[face_i] = false;
        else lut_sense[face_i] = true;
    }

    // Se prepara el punto 3D del centro de la cara
    X0[0] = mp[0];
    X0[1] = mp[1];
    X0[2] = mp[2];

    // Se prepara el sentido del vector normal
    U0[1] = (Nv[1]*sense_factor_int);
    U0[0] = (Nv[0]*sense_factor_int);
    U0[2] = (Nv[2]*sense_factor_int);
    
    // Se recoge la magnitud del vector normal
    std::vector<double> EX0(lut_EX0[face_i]);
    
    // Se prepara el resultado
    result.push_back(X0);
    result.push_back(U0);
    result.push_back(EX0);

    return result;
}

/**
 * @brief Simula el recorrido de un electrón dentro del mallado.
 * 
 * Partiendo de ciertos valores que entran por parámetro, se simula el recorrido de un electrón dentro del mallado.
 * Se incializan ciertos valores, y la función entra en un bucle que simula en cada instante de tiempo la fuerza electromagnética sufrida por el electrón, 
 * y con ello actualiza su velocidad, posición y fase. Comprueba si el electrón sigue dentro del mallado,
 * es decir, si no ha colisionado, y si no ha chocado, sigue la śimulación hasta alcanzar el tiempo límite. 
 * 
 * @param electron_energy double que representa la energía con la que comienza el electrón. Si no se da un valor toma uno por defecto (energy_0) 
 * @param power double que representa la fuerza del dispositivo representado en el mallado en W. Valor por defecto es 1.0
 * @param phase double que representa la fase del electrón al inicio en rad. Valor por defecto 0.0
 * @param face_i int que representa la cara del mallado desde la cual sale el electrón. Valor por defecto es -1, en cuyo caso se escoge una cara al azar
 * @param keep bool que indica si debe guardarse la trayectoria y las energías que ha ido tomando el electrón. Valor por defecto es false
 * @param show_in bool que indica si debe mostrarse la trayectoria del electrón en una representación 3D. Valor por defecto es false, no está implementado en esta versión
 * @return std::tuple<unsigned int, double, double, std::vector<std::vector<double>>> Contiene la cara con la que ha colisionado el electrón, la energía con la que ha colisionado, la fase con la que ha colisionado, y la lista con la trayectoria del electrón.
 *         En caso de que no haya colisionado, la cara tomará el valor 0, la energía será la energía del electrón como si hubiese colisionado y la fase será la fase del electrón en su último instante.
 *         La trayectoria será una lista vacía a no ser que show_in ó keep sean true
 */
std::tuple<unsigned int, double, double, std::vector<std::vector<double>>> track_1_e (double electron_energy = -1, double power = 1.0, double phase = 0.0, int face_i = -1, bool keep = false, bool show_in = false){

    if (log_exec)
    {
        std::ofstream logfile(logfile_name);
        if (!logfile.is_open()) {std::cerr << "2. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
        logfile << "Call track_1_e starting from face= " << face_i << ", phase= " << phase << ", energy=" << electron_energy << std::endl;
        logfile.close();
    }    

    double field_factor = std::sqrt(power);
    std::vector<std::vector<double>> trayectoria;
    std::vector<double> energia_electron;
    unsigned int collision = 0;
    double energy_collision;
    double phase_collision;

    // Si no hay una energía en los parámetros, se usa la energía por defecto
    if (electron_energy == -1) electron_energy = energy_0;

    double max_gamma = 0;
    if (show_in) keep = true;

    // Si no hay una cara por la que empezar, se elige una al azar
    if (face_i == -1) face_i = rand() % N_ext;

    // Se obtienen y se separan las condiciones iniciales de esa cara
    std::vector<std::vector<double>> result = get_initial_conditions_face(face_i);
    std::vector<double> X0  = result[0];
    std::vector<double> U0  = result[1];
    std::vector<double> EX0 = result[2];

    // La velocidad inicial se calcula según la energía del electrón inicial y se guarda en V0
    double gamma = 1.0 + (electron_energy*electron_e_over_mc2);
    double beta  = std::sqrt((gamma*gamma)-1)/gamma;
    double v0    = beta * speed_light_vacuum;
    std::vector<double> V0 = {U0[0]*v0, U0[1]*v0, U0[2]*v0};

    // La fuerza sufrida por el campo electromagnético se guarda en EX
    double t = 0;
    double w = angular_frequency;
    double alpha = (w*t) + phase;
    std::vector<double> EX(3);
    for (int i=0;i<3;i++) EX[i] = (EX0[i]*field_factor*std::cos(alpha));

    // Se inicializan variables para la simulación
    bool ended = false;
    std::vector<double> VX(V0);
    std::vector<double> X(X0);

    // Se calcula el tiempo máximo de simulación
    double t_max = N_cycles/RF_frequency;

    std::vector<double> P0 = {(gamma/speed_light_vacuum)*V0[0],(gamma/speed_light_vacuum)*V0[1],(gamma/speed_light_vacuum)*V0[2]};

    int iteration = 0;
    while (!ended)
    {
        // Para hacer comparaciones de speed-ups, se guarda la cantidad de pasos simulados
        pasos_simulados++;

        // EX0 guarda la fuerza sufrida por el campo electromagnético en el punto del electrón, pero sin tener en cuenta otros factores
        EX0 = evaluate_campo_from_point(X[0],X[1],X[2]);
        alpha = (w*t) + phase;

        // Se actualiza la fuerza sufrida por el electrón
        for (int i=0;i<3;i++) EX[i] = (EX0[i]*field_factor*std::cos(alpha));

        // El cálculo de la velocidad del electrón se hace de manera diferente dependiendo de si el campo electromagnético está encendido o no
        if (!magnetic_field_on)
        {
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

        // Se actualiza la posición en X y se tiene la nueva velocidad como la vieja para la siguiente iteración
        for (int i=0;i<3;i++)
        {
            X[i] = X[i] + VX[i]*delta_t;
            V0[i] = VX[i];
        }

        // Se comprueba si la nueva posición del electrón sigue dentro del mallado
        if (!point_inside_mesh(X))
        {
            // Si se ha salido del mallado, hace falta calcular la cara con la que ha colisionado, la energía y su fase
            double energia_eV = (gamma-1)/electron_e_over_mc2;

            if (gamma <= 1.0)
            {
                std::cerr << "gamma= " << gamma << " eV= " << energia_eV << std::endl;
                std::cerr << VX[0] << " " << VX[1] << " " << VX[2] << " " << std::endl;
                std::exit(1);
            }

            // Se calcula la cara con la que ha colisionado
            dolfin::Point point_x(X[0],X[1],X[2]);
            std::pair<unsigned int, double> collision_face_pair = btree->compute_closest_entity(point_x);

            // En caso de que el cálculo no haya sido correcto, se desplaza la posición ligeramente y se repite el cálculo hasta encontrar un resultado correcto 
            while (collision_face_pair.first >= N_ext)
            {
                std::cerr << "Fallo de btree, con X={" <<X[0]<<","<< X[1]<<","<<X[2]<<"} y collision= " << collision_face_pair.first << "," << collision_face_pair.second << std::endl;
                int xyz = rand() % 3;
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

            // Se preparan los datos para devolverlos en la función
            collision = collision_face_pair.first;
            energy_collision = energia_eV;
            phase_collision = alpha;
            ended = true;
        }

        // Si debe guardarse información de la simulación, se guarda el punto recorrido y la energía que llevaba en dicho punto
        if (keep)
        {
            double energia_eV=(gamma-1)/electron_e_over_mc2;
            trayectoria.push_back(X);
            energia_electron.push_back(energia_eV);
        }

        // Se incrementa el tiempo de simulación
        t = t + delta_t;
        if (t > t_max) {ended = true; std::cerr << "Se acabó el tiempo!" << std::endl;}

    } /* Termina el while not ended*/

    // Si debe guardarse información de la simulación, se escriben en ficheros la trayectoria y la energía que ha ido tomando el electrón
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

    // Si debe motinorizarse la simulación, se escribe en fichero con qué cara ha colisionado el electrón
    if (log_exec)
    {
        std::ofstream logfile(logfile_name);
        if (!logfile.is_open()) {std::cerr << "6. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
        logfile << "Completed, collision= " << collision << std::endl; 
        logfile.close();
    }

    // Si no ha colisionado, significa que se acabó el tiempo de simulación
    if (collision == 0)
    {
        // Se asignan valores de energía 
        double energia_eV=(max_gamma-1)/electron_e_over_mc2;
        energy_collision = energia_eV;
    }

    return std::make_tuple(collision,energy_collision,phase_collision,trayectoria);
}

/**
 * @brief Simula el recorrido de un electrón dentro del mallado y registra el tiempo necesario
 * 
 * @param power double que indica la fuerza del dispositivo representado por el mallado en W 
 * @param face int que representa la cara desde la cual saldrá el electrón
 * @param rf_phase double que representa con qué fase comienza el electrón su simulación
 * @param energy_inic double que representa la energía con la que comienza el electrón su simulación
 * @param keep bool que indica si debe guardarse la trayectoria y energía que va tomando del electrón
 * @return std::tuple<unsigned int, double, double, std::vector<std::vector<double>>> Contiene la cara con la que ha colisionado el electrón, la energía con la que ha colisionado, la fase con la que ha colisionado, y la lista con la trayectoria del electrón.
 *         En caso de que no haya colisionado, la cara tomará el valor 0, la energía será la energía del electrón como si hubiese colisionado y la fase será la fase del electrón en su último instante.
 *         La trayectoria será una lista vacía a no ser que show_in ó keep sean true
 */
std::tuple<unsigned int, double, double, std::vector<std::vector<double>>> run_1_electron(double power, int face, double rf_phase, double energy_inic, bool keep = false)
{
    unsigned int collision;
    double energy_collision, phase_collision;
    std::vector<std::vector<double>> trayectoria;

    // Se toma la muestra de tiempo antes de simular un electrón
    std::chrono::steady_clock::time_point start_1e = std::chrono::steady_clock::now();

    std::tie(collision, energy_collision, phase_collision, trayectoria) = track_1_e(energy_inic,power,rf_phase,face,keep);

    // Se toma la siguiente toma de tiempo después de simular el electrón
    std::chrono::steady_clock::time_point end_1e = std::chrono::steady_clock::now();

    // Se calcula el tiempo que ha tomado
    int segundos_1e     = std::chrono::duration_cast<std::chrono::seconds>(end_1e - start_1e).count();
    int milisegundos_1e = std::chrono::duration_cast<std::chrono::milliseconds>(end_1e - start_1e).count();

    // Se escribe en el fichero el tiempo necesario
    std::ofstream logtime(logtime_name, std::ios::app);
    if (!logtime.is_open()) {std::cerr << "11. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
    if (segundos_1e != 0) logtime << "time 1e: " << segundos_1e << "." << milisegundos_1e << " sec" << std::endl;
    else logtime << "time 1e: " << milisegundos_1e << " ms" << std::endl;
    logtime.close();

    return std::make_tuple(collision,energy_collision,phase_collision,trayectoria);
}

/**
 * @brief Simula el recorrido de un electrón dentro del mallado.
 * 
 * Partiendo de ciertos valores que entran por parámetro, se simula el recorrido de un electrón dentro del mallado.
 * Se incializan ciertos valores, y la función entra en un bucle que simula en cada instante de tiempo la fuerza electromagnética sufrida por el electrón, 
 * y con ello actualiza su velocidad, posición y fase. Comprueba si el electrón sigue dentro del mallado,
 * es decir, si no ha colisionado, y si no ha chocado, sigue la śimulación hasta alcanzar el tiempo límite. 
 * Recibe varios objetos de la librería Dolfin para hacer el cálculo sobre ellos en lugar de sobre los globales.  
 * Esta función se utiliza como alternativa thread-safe.
 * 
 * @param private_pasos_simulados unsigned long long int que cuenta cuántos pasos de electrones se han simulado
 * @param electron_energy double que representa la energía con la que comienza el electrón. Si no se da un valor toma uno por defecto (energy_0) 
 * @param power double que representa la fuerza del dispositivo representado en el mallado en W. Valor por defecto es 1.0
 * @param phase double que representa la fase del electrón al inicio en rad. Valor por defecto 0.0
 * @param face_i int que representa la cara del mallado desde la cual sale el electrón. Valor por defecto es -1, en cuyo caso se escoge una cara al azar
 * @param keep bool que indica si debe guardarse la trayectoria y las energías que ha ido tomando el electrón. Valor por defecto es false
 * @param show_in bool que indica si debe mostrarse la trayectoria del electrón en una representación 3D. Valor por defecto es false, no está implementado en esta versión
 * @param bmesh_in BoundaryMesh mallado exterior 
 * @param btree_in BoundingBoxTree usado para comprobar colisiones y distancias con las entidades más cercanas a un punto
 * @param campoEx_in Función para evaluar la fuerza que ejerce el campo electromagnético en el eje X de un punto
 * @param campoEy_in Función para evaluar la fuerza que ejerce el campo electromagnético en el eje Y de un punto
 * @param campoEz_in Función para evaluar la fuerza que ejerce el campo electromagnético en el eje Z de un punto
 * @param tree_in BoundingBoxTree usado para comprobar colisiones y distancias con las entidades más cercanas a un punto
 * 
 * @return std::tuple<unsigned int, double, double, std::vector<std::vector<double>>> Contiene la cara con la que ha colisionado el electrón, la energía con la que ha colisionado, la fase con la que ha colisionado, y la lista con la trayectoria del electrón.
 *         En caso de que no haya colisionado, la cara tomará el valor 0, la energía será la energía del electrón como si hubiese colisionado y la fase será la fase del electrón en su último instante.
 *         La trayectoria será una lista vacía a no ser que show_in ó keep sean true
 */
std::tuple<unsigned int, double, double, std::vector<std::vector<double>>> track_1_e_thread (unsigned long long int &private_pasos_simulados, double electron_energy = -1, double power = 1.0, double phase = 0.0, int face_i = -1, bool keep = false, bool show_in = false, std::shared_ptr<dolfin::BoundaryMesh> bmesh_in = nullptr,std::shared_ptr<dolfin::BoundingBoxTree> btree_in = nullptr, std::shared_ptr<dolfin::Function> campoEx_in = nullptr, std::shared_ptr<dolfin::Function> campoEy_in = nullptr, std::shared_ptr<dolfin::Function> campoEz_in = nullptr, std::shared_ptr<dolfin::BoundingBoxTree> tree_in = nullptr){

    // Se hacen copias de variables globales
    double private_electron_energy = electron_energy;
    bool private_keep = keep;
    int private_face_i = face_i;
    bool private_show_in = show_in;

    if (log_exec)
    {
        std::ofstream logfile(logfile_name);
        if (!logfile.is_open()) {std::cerr << "2. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
        logfile << "Call track_1_e starting from face= " << face_i << ", phase= " << phase << ", energy=" << private_electron_energy << std::endl;
        logfile.close();
    }

    double field_factor = std::sqrt(power);
    
    std::vector<std::vector<double>> trayectoria;
    std::vector<double> energia_electron;
    unsigned int collision = 0;
    double energy_collision;
    double phase_collision;
    
    // Si no hay una energía en los parámetros, se usa la energía por defecto
    if (private_electron_energy == -1) private_electron_energy = energy_0;

    double max_gamma = 0;
    if (private_show_in) private_keep = true;

    // Si no hay una cara por la que empezar, se elige una al azar
    if (private_face_i == -1) private_face_i = rand() % N_ext;

    // Se obtienen y se separan las condiciones iniciales de esa cara
    auto result = get_initial_conditions_face_thread(private_face_i,bmesh_in,tree_in);
    std::vector<double> X0  = result[0];
    std::vector<double> U0  = result[1];
    std::vector<double> EX0 = result[2];
    
    // La velocidad inicial se calcula según la energía del electrón inicial y se guarda en V0
    double gamma = 1.0 + (private_electron_energy*electron_e_over_mc2);
    double beta  = std::sqrt((gamma*gamma)-1)/gamma;
    double v0    = beta * speed_light_vacuum;
    std::vector<double> V0 = {U0[0]*v0, U0[1]*v0, U0[2]*v0};

    // La fuerza sufrida por el campo electromagnético se guarda en EX
    double t = 0;
    double w = angular_frequency;
    double alpha = (w*t) + phase;
    std::vector<double> EX(3);
    for (int i=0;i<3;i++) EX[i] = (EX0[i]*field_factor*std::cos(alpha));

    // Se inicializan variables para la simulación
    bool ended = false;
    std::vector<double> VX(V0);
    std::vector<double> X(X0);
    
    // Se calcula el tiempo máximo de simulación
    double t_max = N_cycles/RF_frequency;

    std::vector<double> P0 = {(gamma/speed_light_vacuum)*V0[0],(gamma/speed_light_vacuum)*V0[1],(gamma/speed_light_vacuum)*V0[2]};
    
    while (!ended)
    {        
        // Para hacer comparaciones de speed-ups, se guarda la cantidad de pasos simulados
        private_pasos_simulados++;
        
        // EX0 guarda la fuerza sufrida por el campo electromagnético en el punto del electrón, pero sin tener en cuenta otros factores
        EX0 = evaluate_campo_from_point_thread(X[0],X[1],X[2],campoEx_in, campoEy_in, campoEz_in);
        alpha = (w*t) + phase;
        
        // Se actualiza la fuerza sufrida por el electrón
        for (int i=0;i<3;i++) EX[i] = (EX0[i]*field_factor*std::cos(alpha));  

        // El cálculo de la velocidad del electrón se hace de manera diferente dependiendo de si el campo electromagnético está encendido o no
        if (!magnetic_field_on)
        {
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
            std::vector<double> BX = evaluate_campo_from_point_thread(X[0],X[1],X[2],campoEx_in, campoEy_in, campoEz_in);
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

        // Se actualiza la posición en X y se tiene la nueva velocidad como la vieja para la siguiente iteración
        for (int i=0;i<3;i++)
        {
            X[i] = X[i] + VX[i]*delta_t;
            V0[i] = VX[i];
        }

        // Se comprueba si la nueva posición del electrón sigue dentro del mallado
        if (!point_inside_mesh_thread(X,tree_in))
        {
            // Si se ha salido del mallado, hace falta calcular la cara con la que ha colisionado, la energía y su fase
            double energia_eV = (gamma-1)/electron_e_over_mc2;
            if (gamma <= 1.0)
            {
                std::cerr << "gamma= " << gamma << " eV= " << energia_eV << std::endl;
                std::cerr << VX[0] << " " << VX[1] << " " << VX[2] << " " << std::endl;
                std::exit(1);
            }
            
            // Se calcula la cara con la que ha colisionado
            dolfin::Point point_x(X[0],X[1],X[2]);
            std::pair<unsigned int, double> collision_face_pair = btree_in->compute_closest_entity(point_x);
            
            // En caso de que el cálculo no haya sido correcto, se desplaza la posición ligeramente y se repite el cálculo hasta encontrar un resultado correcto 
            while (collision_face_pair.first >= N_ext)
            {
                std::cerr << "Fallo de btree, con X={" <<X[0]<<","<< X[1]<<","<<X[2]<<"} y collision= " << collision_face_pair.first << "," << collision_face_pair.second << std::endl;
                int xyz = rand() % 3;
                double sr = (double) rand()/RAND_MAX;
                sr = sr*0.002 - 0.001;
                std::vector<double> Y = {0,0,0};
                Y[xyz] = sr;
                X[0] += Y[0];
                X[1] += Y[1];
                X[2] += Y[2];
                dolfin::Point point_x(X[0],X[1],X[2]);
                collision_face_pair = btree_in->compute_closest_entity(point_x);
            }

            // Se preparan los datos para devolverlos en la función
            collision = collision_face_pair.first;
            energy_collision = energia_eV;
            phase_collision = alpha;
            ended = true;       
        }
        
        // Si debe guardarse información de la simulación, se guarda el punto recorrido y la energía que llevaba en dicho punto
        if (private_keep)
        {
            double energia_eV=(gamma-1)/electron_e_over_mc2;
            trayectoria.push_back(X);
            energia_electron.push_back(energia_eV);
        }

        // Se incrementa el tiempo de simulación
        t = t + delta_t;
        if (t > t_max) {ended = true; std::cerr << "Se acabó el tiempo!" << std::endl;}

    } /* Termina el while not ended*/

    // Si debe guardarse información de la simulación, se escriben en ficheros la trayectoria y la energía que ha ido tomando el electrón
    if (private_keep)
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

    // Si debe motinorizarse la simulación, se escribe en fichero con qué cara ha colisionado el electrón
    if (log_exec)
    {
        std::ofstream logfile(logfile_name);
        if (!logfile.is_open()) {std::cerr << "6. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
        logfile << "Completed, collision= " << collision << std::endl; 
        logfile.close();
    }
    
    // Si no ha colisionado, significa que se acabó el tiempo de simulación
    if (collision == 0)
    {
        double energia_eV=(max_gamma-1)/electron_e_over_mc2;
        energy_collision = energia_eV;
    }
    return std::make_tuple(collision,energy_collision,phase_collision,trayectoria);
}

/**
 * @brief Simula el recorrido de un electrón dentro del mallado y registra el tiempo necesario.
 * 
 * Recibe varios objetos de la librería Dolfin para hacer el cálculo sobre ellos en lugar de sobre los globales.  
 * Esta función se utiliza como alternativa thread-safe.
 * 
 * @param private_pasos_simulados unsigned long long int que cuenta cuántos pasos de electrones se han simulado
 * @param power double que indica la fuerza del dispositivo representado por el mallado en W 
 * @param face int que representa la cara desde la cual saldrá el electrón
 * @param rf_phase double que representa con qué fase comienza el electrón su simulación
 * @param energy_inic double que representa la energía con la que comienza el electrón su simulación
 * @param keep bool que indica si debe guardarse la trayectoria y energía que va tomando del electrón
 * @param bmesh_in BoundaryMesh mallado exterior 
 * @param btree_in BoundingBoxTree usado para comprobar colisiones y distancias con las entidades más cercanas a un punto
 * @param campoEx_in Función para evaluar la fuerza que ejerce el campo electromagnético en el eje X de un punto
 * @param campoEy_in Función para evaluar la fuerza que ejerce el campo electromagnético en el eje Y de un punto
 * @param campoEz_in Función para evaluar la fuerza que ejerce el campo electromagnético en el eje Z de un punto
 * @param tree_in BoundingBoxTree usado para comprobar colisiones y distancias con las entidades más cercanas a un punto
 * 
 * @return std::tuple<unsigned int, double, double, std::vector<std::vector<double>>> Contiene la cara con la que ha colisionado el electrón, la energía con la que ha colisionado, la fase con la que ha colisionado, y la lista con la trayectoria del electrón.
 *         En caso de que no haya colisionado, la cara tomará el valor 0, la energía será la energía del electrón como si hubiese colisionado y la fase será la fase del electrón en su último instante.
 *         La trayectoria será una lista vacía a no ser que show_in ó keep sean true
 */
std::tuple<unsigned int, double, double, std::vector<std::vector<double>>> run_1_electron_thread(unsigned long long int &private_pasos_simulados, double power, int face, double rf_phase, double energy_0_copy, bool keep = false,std::shared_ptr<dolfin::BoundaryMesh> bmesh_in = nullptr,std::shared_ptr<dolfin::BoundingBoxTree> btree_in = nullptr, std::shared_ptr<dolfin::Function> campoEx_in = nullptr, std::shared_ptr<dolfin::Function> campoEy_in = nullptr, std::shared_ptr<dolfin::Function> campoEz_in = nullptr, std::shared_ptr<dolfin::BoundingBoxTree> tree_in = nullptr)
{
    unsigned int collision;
    double energy_collision, phase_collision;
    std::vector<std::vector<double>> trayectoria;

    // Se toma la muestra de tiempo antes de simular un electrón
    std::chrono::steady_clock::time_point start_1e = std::chrono::steady_clock::now();

    std::tie(collision, energy_collision, phase_collision, trayectoria) = track_1_e_thread(private_pasos_simulados,energy_0_copy,power,rf_phase,face,keep,false,bmesh_in,btree_in,campoEx_in,campoEy_in,campoEz_in,tree_in);

    // Se toma la siguiente toma de tiempo después de simular el electrón
    std::chrono::steady_clock::time_point end_1e = std::chrono::steady_clock::now();

    // Se calcula el tiempo que ha tomado
    int segundos_1e     = std::chrono::duration_cast<std::chrono::seconds>(end_1e - start_1e).count();
    int milisegundos_1e = std::chrono::duration_cast<std::chrono::milliseconds>(end_1e - start_1e).count();

    // Se escribe en el fichero el tiempo necesario
    std::ofstream logtime(logtime_name, std::ios::app);
    if (!logtime.is_open()) {std::cerr << "11. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
    if (segundos_1e != 0) logtime << "time 1e: " << segundos_1e << "." << milisegundos_1e << " sec" << std::endl;
    else logtime << "time 1e: " << milisegundos_1e << " ms" << std::endl;
    logtime.close();

    return std::make_tuple(collision,energy_collision,phase_collision,trayectoria);
}

/**
 * @brief Simula varios electrones dentro del mallado
 * 
 * Esta función parte de una lista que contiene uno o varios electrones, y sus listas complementarias que contienen las energías y fases iniciales de cada electrón.
 * Despues, simula los electrones iniciales y añade los electrones resultantes por la colisión a la siguiente 'generación'. Repite este proceso hasta alcanzar el número
 * de generaciones máximo. La simulación de los electrones de cada generación se distribuye entre todos los threads disponibles según scheduling guiado.
 *
 * @param power double que representa la fuerza del dispositivo representado con el mallado
 * @param pool_runs lista de integers que contiene las caras por las cuales se lanzan los electrones a simular
 * @param pool_phase lista de doubles que contiene las fases iniciales de los electrones a simulas 
 * @param pool_energies lista de doubles que contiene las energías iniciales de los electrones a simulas  
 * @return std::tuple<int, int> Representan la cantidad de electrones que se han simulado en total y la cantidad de electrones que se han simulado en la última generación.
 */
std::tuple<int, int> run_n_electrons_parallel (double power, Electron_list& pool_runs){

    // Mostrar y guardar estado inicial
    std::cout << "Power=" << power << "W, initial #electrons: " << pool_runs.electron_count << std::endl;
    std::ofstream logtime(logtime_name, std::ios::app);
    if (!logtime.is_open()) {std::cerr << "9. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
    logtime << "Power=" << power << "W, initial #electrons: " << pool_runs.electron_count << std::endl;
    logtime.close();

    // Si debe monitorizarse la ejecución, se guardan las condiciones iniciales en fichero
    if (log_exec)
    {
        std::ofstream logfile(logfile_name, std::ios::app);
        if (!logfile.is_open()) {std::cerr << "10. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
        logfile << "Power=" << power << "W, initial #electrons: " << pool_runs.electron_count << std::endl;
        logfile.close();
    }

    int n = 0, number_electrons = 0;
    bool ended = false;
    std::chrono::steady_clock::time_point start_run, end_run;

    // Se crea la lista idéntica a la de entrada para las siguientes generaciones
    Electron_list new_pool_runs;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int tnr = omp_get_num_threads();

        // Variables privadas para cada thread, son copias de las globales
        std::shared_ptr<dolfin::BoundingBoxTree> tree_private = std::make_shared<dolfin::BoundingBoxTree>(*tree);
        std::shared_ptr<dolfin::BoundingBoxTree> btree_private = std::make_shared<dolfin::BoundingBoxTree>(*btree);
        std::shared_ptr<dolfin::BoundaryMesh> bmesh_private = std::make_shared<dolfin::BoundaryMesh>(*bmesh);
        std::shared_ptr<dolfin::Function> campoEx_private = std::make_shared<dolfin::Function>(*campoEx);
        std::shared_ptr<dolfin::Function> campoEy_private = std::make_shared<dolfin::Function>(*campoEy);
        std::shared_ptr<dolfin::Function> campoEz_private = std::make_shared<dolfin::Function>(*campoEz);

        Electron_list private_new_pool_runs;

        // Cada thread cuenta cuántos pasos ha simulado
        unsigned long long int private_pasos_simulados=0;
        
        // Bucle que se repite hasta que una generación a simular este vacía o se haya alcanzado el máximo número de generaciones
        while (!ended)
        {
            #pragma omp master
            { 
                // Se toma la muestra de tiempo antes de simular la generación
                start_run = std::chrono::steady_clock::now();

                // Si la generación está vacía, se sale del bucle
                if (pool_runs.electron_count == 0)
                {
                    ended = true;
                }
            } 

            #pragma omp barrier  

            if (ended)  continue;         

            unsigned int face;
            int erun;
            double efase, energy_0_copy, energy, phase;
            std::vector<std::vector<double>> trayectoria;
            
            // Para cada electrón a simular en la generación actual
            std::shared_ptr<Electron> actual_electron = std::make_shared<Electron>(*get_electron(pool_runs));
            while ( actual_electron->face != -1)
            {
                // Se toman los valores iniciales
                erun = actual_electron->face;
                efase = actual_electron->phase;
                energy_0_copy = actual_electron->energy;

                // Se simula el electrón
                std::tie(face, energy, phase, trayectoria) = run_1_electron_thread(private_pasos_simulados,power,erun,efase,energy_0_copy,false,bmesh_private,btree_private,campoEx_private,campoEy_private,campoEz_private,tree_private);
            
                // Si ha colisionado
                if (face != 0) 
                {
                    int n_e; std::vector<double> energies; double sey;

                    // Se calculan los electrones secundarios generados y sus energías
                    std::tie(n_e,energies,sey) = total_secondary_electrons(energy);
                    
                    // Se añaden los electrones nuevos, sus energías y fases a la siguiente generación
                    for (int j=0;j<n_e;j++)
                    {
                        std::shared_ptr<Electron> new_e = std::make_shared<Electron>();
                        new_e->face   = face;
                        new_e->phase  = phase;
                        new_e->energy = energies[j];
                        new_e->next   = nullptr;

                        // Se va construyendo la lista con los electrones nuevos
                        add_electron(private_new_pool_runs,new_e); 
                    }

                    // Se añaden a la lista de la siguiente generación los nuevos electrones
                    append_list(new_pool_runs,private_new_pool_runs);

                    // Se limpia la lista de los electrones generados para simular el siguiente
                    resetear_lista(private_new_pool_runs);
                }

                // Se busca el siguiente electrón de la lista
                actual_electron = std::make_shared<Electron>(*get_electron(pool_runs));
            }
            
            // Es necesario que la siguiente generación esté completa antes de continuar
            #pragma omp barrier
            
            #pragma omp master
            { 
                // El procesador principal asigna la siguiente generación como la actual para la siguiente iteración
                copiar_lista(pool_runs,new_pool_runs);

                // Se limpia la lista de la siguiente generación
                resetear_lista(new_pool_runs);

                // Guardar tiempo de una 'run'
                end_run = std::chrono::steady_clock::now();

                double macrosec_run = std::chrono::duration_cast<std::chrono::microseconds>(end_run - start_run).count();
                double minutos_run = macrosec_run/60000000;

                // Guardar en fichero el tiempo de simular una generación y la cantidad de electrones por simular 
                std::ofstream logtime(logtime_name, std::ios::app);
                if (!logtime.is_open()) {std::cerr << "13. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
                if (minutos_run >= 1) logtime << "Completed run " << n << ", time: " << minutos_run << " min, electrons alive: " << pool_runs.electron_count << std::endl;
                else logtime << "Completed run " << n << ", time: " << minutos_run*60 << " sec, electrons alive: " << pool_runs.electron_count << std::endl;
                logtime.close();

                // Mostrar tiempo de una 'run'
                std::cout << "Time of run: " << std::setprecision(2) << minutos_run << " min" << std::endl;
                std::cout << "Power=" << power << " W, run#: " << n << ", electrons alive:" << pool_runs.electron_count << std::endl;

                // Si se debe monitorizar la ejecución, se guarda lo mismo en fichero
                if (log_exec)
                {
                    std::ofstream logfile(logfile_name, std::ios::app);
                    if (!logfile.is_open()) {std::cerr << "14. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
                    logfile << "Completed secondary run #: " << n << ", power=" << power << " W, electrons alive:" << pool_runs.electron_count << std::endl << std::endl;
                    logfile.close();
                }

                // Actualizar número total de electrones simulados y generaciones
                number_electrons = number_electrons + pool_runs.electron_count;
                n = n+1;

                // Comprobar si se ha alcanzado el número máximo de generaciones a simular
                if (n>N_max_secondary_runs)
                {
                    ended = true;
                    std::cout << "Max number of secondary runs achieved at P=" << power << " W" << std::endl;
                    
                    // Si se debe monitorizar la ejecución, guardar un aviso en fichero
                    if (log_exec)
                    {
                        std::ofstream logfile(logfile_name, std::ios::app);
                        if (!logfile.is_open()) {std::cerr << "15. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
                        logfile << "Max number of secondary runs achieved at P=" << power << " W" << std::endl;
                        logfile.close();
                    }   
                }
            }

            // Todos los procesadores deben tener la lista completa de la siguiente generación antes de continuar
            #pragma omp barrier
        }
    
        // Al final de la ejecución paralela, se suman los pasos que ha simulado cada thread
        #pragma omp critical (pasos_simulados)
        {
            pasos_simulados += private_pasos_simulados;
        }
    }

    return std::make_tuple(number_electrons,pool_runs.electron_count);
}

/**
 * @brief Inicia el programa
 * 
 * Esta función genera los ficheros de logging y lanza la simulación, dependiendo del modo de simulación 
 * 1 -> Simular efecto multipactor en varias configuraciones de power
 * 2 -> Simular la trayectoria de un sólo electrón
 * 3 -> Simular efecto multipactor en un único valor de power
 *
 * @return int que indica el fin de ejecución de la función
 */
int run(){

    // Se toma la muestra de tiempo para calcular el tiempo total de la simulación
    time_t start_total = std::time(nullptr);

    // Si no se ha dado una semilla para el motor random, se utiliza el tiempo actual
    std::cout << "random_seed: " << random_seed << std::endl;
    if (random_seed == -1) srand (time(NULL));
    else srand(random_seed);

    // Se representa el tiempo actual de ejecución en string
    tm start_total_struct = *std::localtime(&start_total);
    std::ostringstream oss;
    oss << std::put_time(&start_total_struct, "%Y%m%d_%H%M%S");
    std::string now_str = oss.str();

    // Se guarda el nombre del fichero usado para logging
    logfile_name = "generated_files/log_mpc_py_" + now_str + ".txt";

    // Si ha de simularse el efecto multipactor y debe monitorizarse la simulación, se crea el fichero de logging
    if (log_exec && (simulation_type != 2))
    {
        std::ofstream logfile(logfile_name);
        if (!logfile.is_open()) {std::cerr << "7. Error opening log file: " << logfile_name << std::endl; std::exit(1);}
        logfile.close();
    }

    // Se crea el fichero para registrar los tiempos de simulación
    logtime_name = "generated_files/exec_time_" + now_str + ".txt";
    std::ofstream logtime(logtime_name);
    if (!logtime.is_open()) {std::cerr << "8. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
    logtime.close();
    
    // Si ha de simularse la trayectoria de un solo electrón
    if (simulation_type == 2)
    {
        // Se repite el bucle hasta encontrar una cara válida para el inicio
        std::vector<int> test_face;
        while (test_face.empty())
        {
            // Se añade una cara aleatoria
            test_face.push_back(rand() % N_ext);

            // Se repasan las condiciones para poner a prueba la cara
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

        // Se simula el electrón
        std::tie(collision, energy_collision, phase_collision, trayectoria) = track_1_e(-1.0,RF_power[0],0.0,-1,true);

        // Termina la función
        return 0;
    } 
    // Si debe simularse solo un power, comienza con 1 electrón
    else if (simulation_type == 3) N_runs_per_power = 1;
    // Si hay más de un power, cada uno comienza con 'electrons_seed' electrones
    else if (simulation_type == 1) N_runs_per_power = electrons_seed;
    
    // Se imprimen los valores de power
    std::cout << "---Imprimiendo power---" << std::endl;
    for (auto element : RF_power) std::cout << element << " ";
    std::cout << std::endl;

    std::vector<int> total_electrons, final_electrons, power_partial;
    int number_of_electrons, electrons_last_cycle;

    // Para cada power se simula el efecto multipactor
    for (const auto power : RF_power)
    {
        Electron_list pool_runs;
        std::vector<int> new_pool_runs;

        // Se escogen caras aleatoriamente
        for (int i=0;i<N_runs_per_power;i++) new_pool_runs.push_back(rand() % N_elems); 

        // Se imprimen las caras
        std::cout << "New_pool_runs: ";
        for (auto element : new_pool_runs) std::cout << element << " ";
        std::cout << std::endl;

        // Se someten las caras a las condiciones establecidas
        for (std::vector<double> ct : boundaries_excluded_boolean)
        {
            double operation  = ct[0];
            double coordinate = ct[1];
            double value      = ct[2];
            double tolerance  = ct[3];
            new_pool_runs = remove_by_boolean_condition (new_pool_runs, operation, coordinate, value, tolerance);
        }

        // Se inicializan las energías y fases aleatoriamente
        for (int i=0; i<new_pool_runs.size();i++)
        {
            std::shared_ptr<Electron> new_e = std::make_shared<Electron>();
            double random_double = (double) rand()/RAND_MAX;

            new_e->face   = new_pool_runs[i];
            new_e->phase  = random_double*2.0*boost::math::constants::pi<double>();
            new_e->energy = energy_0;
            new_e->next   = nullptr;

            add_electron(pool_runs,new_e);

        }

        // Se simula el efecto multipactor 
        std::tie(number_of_electrons, electrons_last_cycle) = run_n_electrons_parallel (power, pool_runs);
    
        // Se acumulan los valores obtenidos
        total_electrons.push_back(number_of_electrons);
        final_electrons.push_back(electrons_last_cycle);
        power_partial.push_back(power);
    }

    // Se guarda información de los valores acumulados en fichero
    std::ofstream calculo_multipacing("generated_files/calculo_multipacting.txt");
    if (!calculo_multipacing.is_open()) {std::cerr << "16. Error opening calculo_multipacting.txt file" << std::endl; std::exit(1);}
    for (int i=0;i<total_electrons.size();i++) calculo_multipacing << power_partial[i] << "\t" << total_electrons[i] << "\t" << final_electrons[i] << std::endl;
    calculo_multipacing.close();

    // Mostrar tiempo total
    time_t end_total = std::time(nullptr);
    tm end_total_struct = *std::localtime(&end_total);

    double minutes_total = (end_total_struct.tm_min - start_total_struct.tm_min);
    double seconds_total = (end_total_struct.tm_sec - start_total_struct.tm_sec);
    double hours_total = (end_total_struct.tm_hour - start_total_struct.tm_hour);
    
    minutes_total += (seconds_total/60) + (hours_total*60);
    std::cout << "Total time: " << minutes_total << " min" << std::endl;

    // Guardar tiempo total
    std::ofstream logtime2(logtime_name, std::ios::app);
    if (!logtime2.is_open()) {std::cerr << "17. Error opening logtime file: " << logtime_name << std::endl; std::exit(1);}
    if (minutes_total>=1) logtime2 << "Total execution time: " << std::setprecision(3) << minutes_total << " min" << std::endl;
    else logtime2 << "Total execution time: " << std::setprecision(3) << minutes_total*60 << " sec" << std::endl;
    logtime2.close();

    // El total de pasos simulados sirve para hacer comparaciones con versiones multithreading y/o multiprocessing
    std::cout << "Total de pasos simulados: " << pasos_simulados << std::endl;

    return 0;
}

int main(int argc, char* argv[]) {
    const float VERSION = 1.0;

    if (argc < 2)
    {
        std::cerr << "mpc_run version = " << VERSION << " - Run multipacting calculations (jlmunoz@essbilbao.org)" << std::endl;
        std::cerr << "See problem file (*.mpc) for description of parameters." << std::endl;
        std::cerr << "Syntax: python mpc_run.py problem.mpc" << std::endl;
        std::cerr << "(If you need an example .mpc file, `python mpc_run.py test` will generate test.mpc for you.)" << std::endl;
        std::exit(1);
    }

    std::ifstream mpc_file(argv[1]);

    if (!mpc_file)
    {
        std::cerr << "Error opening the file, ending execution" << std::endl;
        std::exit(1);
    }

    // Obtener los parámetros generales a partir del fichero .mpc
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

    // Inicializar variables según lo obtenido del fichero .mpc
    set_parameters_dictionary();

    // Leer los ficheros del mallado
    read_from_data_files();

    // Ejecutar la simulación
    run();

    return 0;
}
