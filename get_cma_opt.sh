#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --parsable
#SBATCH --mem=64000M
#SBATCH --account=def-rbdong
#SBATCH --job-name=guild_cma_opt
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
echo "$GUILD_HOME"
guild compare -Fo onet:cma_opt -u --csv "$GUILD_HOME"/cma_opt.csv
