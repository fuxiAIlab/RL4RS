#!/bin/bash

conda activate rl4rs
script_abs=$(readlink -f "$0")
rl4rs_benchmark_dir=$(dirname $script_abs)/..
rl4rs_output_dir=${rl4rs_benchmark_dir}/output
rl4rs_dataset_dir=${rl4rs_benchmark_dir}/dataset
script_dir=${rl4rs_benchmark_dir}/script
export rl4rs_benchmark_dir && export rl4rs_output_dir && export rl4rs_dataset_dir

algo=$1

cd ${script_dir}

head -1 ${rl4rs_dataset_dir}/rl4rs_dataset_a_train.csv > ${rl4rs_dataset_dir}/rl4rs_dataset_a_train_tiny.csv
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_a_train_tiny.csv" "${rl4rs_output_dir}/rl4rs_dataset_a_train_tiny.tfrecord" "tfrecord_item"
python -u simulator_env_test.py "{'env':'SlateRecEnv-v0','support_conti_env':False,'rawstate_as_obs':False}" &&
python -u simulator_env_test.py "{'env':'SlateRecEnv-v0','support_conti_env':False,'rawstate_as_obs':True}" &&
python -u simulator_env_test.py "{'env':'SlateRecEnv-v0','support_conti_env':True,'rawstate_as_obs':False,'action_emb_size':32}" &&
python -u simulator_env_test.py "{'env':'SlateRecEnv-v0','support_conti_env':True,'rawstate_as_obs':True,'action_emb_size':32}" &&
echo '1'