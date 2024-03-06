#!/bin/bash



#SBATCH --job-name=swyft_training
#SBATCH --account=ec12
#SBATCH --time=00-00:10:00
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=40G
#SBATCH --qos=devel





set -o errexit  # Exit the script on any error

#set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default

# Set the ${PS1} (needed in the source of the Anaconda environment)

export PS1=\$

module load Miniconda3/22.11.1-1

source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh

conda deactivate &>/dev/null

conda activate /fp/homes01/u01/ec-gertwk/.conda/envs/swyft4-dev






 python configure_obs.py 




exit
