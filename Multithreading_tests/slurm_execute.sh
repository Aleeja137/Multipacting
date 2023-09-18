#!/bin/bash                                                          
#SBATCH --partition=regular
#SBATCH --job-name=mpc_run
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --error=slurm_output/mpc_run_err.txt
#SBATCH --output=slurm_output/mpc_run_out.txt
#SBATCH --mail-user=aperez440@ikasle.ehu.eus
#SBATCH --mail-type=ALL


# modules
conda activate /scratch/aperez440/fenics-env
module load CMake/3.24.3-GCCcore-12.2.0
module load Boost/1.77.0-GCC-11.2.0
module load GCCcore/8.2.0
export CPATH=/scratch/aperez440/fenics-env/include
export CPATH=/scratch/aperez440/fenics-env/include/eigen3:$CPATH
export LD_LIBRARY_PATH=/scratch/aperez440/fenics-env/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/scratch/aperez440/fenics-env/lib/pkgconfig:$PKG_CONFIG_PATH

export OMP_SCHEDULE="guided"
export OMP_NUM_THREADS=4

./mpc_run data/coaxial_704_01.mpc
conda deactivate
