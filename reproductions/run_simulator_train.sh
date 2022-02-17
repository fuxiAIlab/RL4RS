#!/bin/bash

conda activate rl4rs
script_abs=$(readlink -f "$0")
rl4rs_benchmark_dir=$(dirname $script_abs)/..
rl4rs_output_dir=${rl4rs_benchmark_dir}/output
script_dir=${rl4rs_benchmark_dir}/script
export rl4rs_benchmark_dir && export rl4rs_output_dir && export rl4rs_dataset_dir

algo=$1

cd ${script_dir}

# RL Env Construction

python supervised_train.py "${rl4rs_output_dir}/rl4rs_dataset_a_sl.tfrecord" "${rl4rs_output_dir}/rl4rs_dataset_a_rl.tfrecord" "${rl4rs_output_dir}/simulator_a_sl_$algo/model" $algo 0 >> ${rl4rs_output_dir}/simulator_a_sl_${algo}.log &&

python supervised_train.py "${rl4rs_output_dir}/rl4rs_dataset_a_rl.tfrecord" "${rl4rs_output_dir}/rl4rs_dataset_a_sl.tfrecord" "${rl4rs_output_dir}/simulator_a_rl_$algo/model" $algo 0 >> ${rl4rs_output_dir}/simulator_a_rl_${algo}.log &&

python supervised_train.py "${rl4rs_output_dir}/rl4rs_dataset_a.tfrecord" "${rl4rs_output_dir}/rl4rs_dataset_a.tfrecord" "${rl4rs_output_dir}/simulator_a_$algo/model" $algo 0 >> ${rl4rs_output_dir}/simulator_a_${algo}.log &&

python supervised_train.py "${rl4rs_output_dir}/rl4rs_dataset_b2_sl.tfrecord" "${rl4rs_output_dir}/rl4rs_dataset_b2_rl.tfrecord" "${rl4rs_output_dir}/simulator_b2_sl_$algo/model" $algo 0 >> ${rl4rs_output_dir}/simulator_b2_sl_${algo}.log &&

python supervised_train.py "${rl4rs_output_dir}/rl4rs_dataset_b2_rl.tfrecord" "${rl4rs_output_dir}/rl4rs_dataset_b2_sl.tfrecord" "${rl4rs_output_dir}/simulator_b2_rl_$algo/model" $algo 0 >> ${rl4rs_output_dir}/simulator_b2_rl_${algo}.log &&

python supervised_train.py "${rl4rs_output_dir}/rl4rs_dataset_b2.tfrecord" "${rl4rs_output_dir}/rl4rs_dataset_b2.tfrecord" "${rl4rs_output_dir}/simulator_b2_$algo/model" $algo 0 >> ${rl4rs_output_dir}/simulator_b2_${algo}.log &&

echo "1"