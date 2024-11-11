#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --parsable
#SBATCH --mem=64000M
#SBATCH --account=def-rbdong
#SBATCH --job-name=run
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=symao@uvic.ca
#SBATCH --mail-type=ALL

for var_name in "GUILD_HOME" "CMA_VENV"; do
  if [ -z "${!var_name}" ]; then
    echo "The environment variable '$var_name' is not configured or is empty."
    exit 1
  else
    echo "The environment variable '$var_name' has the value: ${!var_name}"
  fi
done
# cd
cd "$ONET_DISK2D_SINGLE_HOME" || exit
# load module
module load StdEnv/2020  gcc/11.3.0 cudacore/.11.7.0
module load cuda/11.7
module load cudnn/8.7
# activate env
source "$CMA_VENV"/bin/activate
# guild run
# monitor gpu usage
# nvidia-smi --query-gpu=timestamp,name,pstate,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used -l 10 --format=csv &
for id in "$@"; do
	echo "id=${id}"
	guild run --start $id -y
done
