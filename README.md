# MPC Port C++ 
Port de código científico python del grupo ESS Bilbao a C++  
Paralelización mediante OpenMP  

# Entorno  
  
## Instalación  
Para el entorno en ATLAS (mediante conda)  
    module load python  
    conda create --prefix /scratch/XXX/fenics-env -c conda-forge fenics  

## Compilación/ejecución  
conda activate /scratch/XXX/fenics-env  
module load CMake/3.24.3-GCCcore-12.2.0  
module load Boost/1.77.0-GCC-11.2.0  
module load GCCcore/8.2.0  
export CPATH=/scratch/XXX/fenics-env/include  
export CPATH=/scratch/XXX/fenics-env/include/eigen3:$CPATH  
export LD_LIBRARY_PATH=/scratch/XXX/fenics-env/lib:$LD_LIBRARY_PATH  
export PKG_CONFIG_PATH=/scratch/XXX/fenics-env/lib/pkgconfig:$PKG_CONFIG_PATH  
  
# Ejecución  
sbatch slurm_compile.sh  
sbatch slurm_execute.sh  
  
slurm_execute.sh ejecuta por defecto data/coaxial_704_01.mpc, que por defecto ejecuta el código en serie.   
    ./mpc_run data/coaxial_704_01.mpc  
  
Los ficheros .mpc disponibles son:  
data/coaxial_704_01.mpc  
data/pikachu_testbox_500k.mpc  
data/pikachu_testbox_900k.mpc (el fichero de mallado pikachu_couplers_testbox_900k.mphtxt.h5 no está incluido en el repositorio)  
  
Cambiando el valor de 'parallel' en los ficheros .mpc es posible ejecutar el código con diferente número de threads  
