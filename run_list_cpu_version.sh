#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --parsable
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
# activate env
source "$CMA_VENV"/bin/activate
# guild run
for id in "$@"; do
	echo "id=${id}"
	guild run --start $id -y
done
