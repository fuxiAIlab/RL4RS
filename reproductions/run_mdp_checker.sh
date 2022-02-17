#!/bin/bash

conda activate rl4rs
script_abs=$(readlink -f "$0")
rl4rs_benchmark_dir=$(dirname $script_abs)/..
rl4rs_output_dir=${rl4rs_benchmark_dir}/output
rl4rs_dataset_dir=${rl4rs_benchmark_dir}/dataset
script_dir=${rl4rs_benchmark_dir}/script
export rl4rs_benchmark_dir && export rl4rs_output_dir && export rl4rs_dataset_dir

dataset=$1

cd ${script_dir}/tool
python -u preprocess.py $dataset ${rl4rs_dataset_dir} &&
python -u mdp_checker.py $dataset ${rl4rs_dataset_dir} >> ${rl4rs_output_dir}/data_understanding_tool_${dataset}.log &&
echo "1"