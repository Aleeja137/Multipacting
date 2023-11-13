La versión en serie del código C++ para simular el efecto multipactor dentro de un coaxial_103mm_704MHz

# Ejemplo de uso
## En portátil
### Requerimientos
- Instalación de FEniCS (preferiblemente usando conda)
- CMake 3.26.4 o superior
- Boost 1.77 o superior
- GCCcore 7.5.0 o superior
- OpenMP

### Compilación
conda activate /path/to/fenics/conda
export CPATH=/path/to/fenics/conda/include
export CPATH=/path/to/fenics/conda/include/eigen3:$CPATH
export LD_LIBRARY_PATH=/path/to/fenics/conda/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/path/to/fenics/conda/lib/pkgconfig:$PKG_CONFIG_PATH

cd build/
cmake ..
make
mv mpc_run ../mpc_run
conda deactivate

### Ejecución
export OMP_SCHEDULE="guided"
export OMP_NUM_THREADS=X (Hasta 8)

./mpc_run data/coaxial_704_1.mpc
(No existe una demo todavía)

### Parámetros
Modificar los parámetros en el fichero data/coaxial_704_1.mpc. 
Especialmente 
- N_max_secondary_runs=60 (determina el número de generaciones máximo)
- simulation_type=3 (determina qué tipo de simulación se ejecuta)
- random_seed=2 (semilla usada para los valores aleatorios, con 2 se consigue un número de electrones decente, con 3 uno muy alto)

## En ATLAS
### Compilación
sbatch slurm_compile.sh

### Ejecución
Modificar data/coaxial_704_1.mpc
Modificar slurm_execute.sh para que el tiempo sea consistente con el número de generaciones, y los hilos
sbatch slurm_execute.sh
(No existe una demo todavía)

### Parámetros
Modificar los parámetros en el fichero data/coaxial_704_1.mpc. 
Especialmente 
- N_max_secondary_runs=60 (determina el número de generaciones máximo)
- simulation_type=3 (determina qué tipo de simulación se ejecuta)
- random_seed=2 (semilla usada para los valores aleatorios, con 2 se consigue un número de electrones decente, con 3 uno muy alto)

Modificar tiempo necesario en slurm_execute.sh
- #SBATCH --cpus-per-task=4 (Para hacer la petición de los threads, 24-..-56)