# export GUILD_HOME=guild_home/cma/sigma
# export DATASET_ID=801a6ff6
# export NETWORK_ID=log_sigma_model
for var_name in "GUILD_HOME" "CMA_NETWORK_ROOT" "NETWORK_ID" "CMA_DATA_ROOT" "DATASET_ID"; do
  if [ -z "${!var_name}" ]; then
    echo "The environment variable '$var_name' is not configured or is empty."
    exit 1
  else
    echo "The environment variable '$var_name' has the value: ${!var_name}"
  fi
done
keys=$(printf "%s," "$@")
keys="${keys%,}"
guild run onet:cma_opt network_root_dir="$CMA_NETWORK_ROOT" network_id="$NETWORK_ID" data_root_dir="$CMA_DATA_ROOT" dataset_id="$DATASET_ID" save_dir=. metric=l2 popsize=128 maxiter=512 key=[$keys] init_sample_m=12 sigma0=0.01 std_alpha=1.0 std_h0=1.0 std_q=1.0 std_r_p=1.0 std_theta_p=1.0 x0_search=sobol_scramble @"$GUILD_HOME"/change_seed.csv --stage-trials -y
guild compare -Fo onet:cma_opt -u --csv "$GUILD_HOME"/cma_opt.csv
