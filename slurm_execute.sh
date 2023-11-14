#!/bin/bash                                                          
#SBATCH --partition=regular
#SBATCH --job-name=mpc_run
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --error=slurm_output/mpc_run_err.txt
#SBATCH --output=slurm_output/mpc_run_out.txt
#SBATCH --mail-user=XXX@YYY.ZZZ
#SBATCH --mail-type=ALL


# modules
conda activate /scratch/XXX/fenics-env
module load CMake/3.24.3-GCCcore-12.2.0
module load Boost/1.77.0-GCC-11.2.0
module load GCCcore/8.2.0
export CPATH=/scratch/XXX/fenics-env/include
export CPATH=/scratch/XXX/fenics-env/include/eigen3:$CPATH
export LD_LIBRARY_PATH=/scratch/XXX/fenics-env/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/scratch/XXX/fenics-env/lib/pkgconfig:$PKG_CONFIG_PATH

./mpc_run data/coaxial_704_01.mpc
conda deactivate
