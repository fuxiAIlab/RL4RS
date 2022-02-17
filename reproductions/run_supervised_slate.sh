#!/bin/bash

conda activate rl4rs
script_abs=$(readlink -f "$0")
rl4rs_benchmark_dir=$(dirname $script_abs)/..
rl4rs_output_dir=${rl4rs_benchmark_dir}/output
script_dir=${rl4rs_benchmark_dir}/script
export rl4rs_benchmark_dir && export rl4rs_output_dir && export rl4rs_dataset_dir

algo=$1

cd ${script_dir}

# supervised learning evaluation

python supervised_train.py "${rl4rs_output_dir}/rl4rs_dataset_a_train_slate.tfrecord" "${rl4rs_output_dir}/rl4rs_dataset_a_test_slate.tfrecord" "${rl4rs_output_dir}/supervised_a_train_slate_$algo/model" $algo 1 >> ${rl4rs_output_dir}/supervised_a_train_${algo}_slate.log &&

python supervised_train.py "${rl4rs_output_dir}/rl4rs_dataset_b2_train_slate.tfrecord" "${rl4rs_output_dir}/rl4rs_dataset_b2_test_slate.tfrecord" "${rl4rs_output_dir}/supervised_b2_train_slate_$algo/model" $algo 1 >> ${rl4rs_output_dir}/supervised_b2_train_${algo}_slate.log &&

echo "1"

