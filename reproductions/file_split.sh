#!/bin/bash

script_abs=$(readlink -f "$0")
rl4rs_benchmark_dir=$(dirname $script_abs)/..
rl4rs_dataset_dir=${rl4rs_benchmark_dir}/dataset
script_dir=${rl4rs_benchmark_dir}/script
rl4rs_output_dir=${rl4rs_benchmark_dir}/output
export rl4rs_benchmark_dir && export rl4rs_output_dir && export rl4rs_dataset_dir

file=$1

cd ${rl4rs_dataset_dir} &&

awk -F "@" '$2%11<2 {print}' ${file} > ${rl4rs_output_dir}/${file}_0000.csv &&
awk -F "@" '$2%11>=2 && $2%11<4 {print}' ${file} > ${rl4rs_output_dir}/${file}_0001.csv &&
awk -F "@" '$2%11>=4 && $2%11<6 {print}' ${file} > ${rl4rs_output_dir}/${file}_0002.csv &&
awk -F "@" '$2%11>=6 && $2%11<8 {print}' ${file} > ${rl4rs_output_dir}/${file}_0003.csv &&
awk -F "@" '$2%11>=8 {print}' ${file} > ${rl4rs_output_dir}/${file}_0004.csv

#file_rows=`wc -l ${file}|awk '{print $1}'`
#file_num=5
#file_num_row=$((${file_rows} + 4))
#every_file_row=$((${file_num_row}/${file_num}))
#split -d -a 4 -l ${every_file_row} ${file} --additional-suffix=.csv ${rl4rs_output_dir}/${file}_


