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

# train in train set and test in all sample
python -u simulator_eval.py "{'env':'SlateRecEnv-v0','algo':'${algo}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_shuf.csv','model_file':'${rl4rs_output_dir}/supervised_a_train_${algo}/model'}" >> ${rl4rs_output_dir}/eval_simulator_a_all_${algo}.log &&
python -u simulator_eval.py "{'gpu':False,'env':'SeqSlateRecEnv-v0','algo':'${algo}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_shuf.csv','model_file':'${rl4rs_output_dir}/supervised_b2_train_${algo}/model'}" >> ${rl4rs_output_dir}/eval_simulator_b2_all_${algo}.log

# train in all set and test in sl/rl as a baseline
python -u simulator_eval.py "{'env':'SlateRecEnv-v0','algo':'${algo}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_sl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_${algo}/model'}" >> ${rl4rs_output_dir}/eval_simulator_a_all_sl_${algo}.log &&
python -u simulator_eval.py "{'env':'SlateRecEnv-v0','algo':'${algo}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_${algo}/model'}" >> ${rl4rs_output_dir}/eval_simulator_a_all_rl_${algo}.log &&

python -u simulator_eval.py "{'gpu':False,'env':'SeqSlateRecEnv-v0','algo':'${algo}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_sl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_${algo}/model'}" >> ${rl4rs_output_dir}/eval_simulator_b2_all_sl_${algo}.log &&
python -u simulator_eval.py "{'gpu':False,'env':'SeqSlateRecEnv-v0','algo':'${algo}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_${algo}/model'}" >> ${rl4rs_output_dir}/eval_simulator_b2_all_rl_${algo}.log &&

# train in sl/rl and test in rl/sl
python -u simulator_eval.py "{'env':'SlateRecEnv-v0','algo':'${algo}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_sl_${algo}/model'}" >> ${rl4rs_output_dir}/eval_simulator_a_sl_rl_${algo}.log &&
python -u simulator_eval.py "{'env':'SlateRecEnv-v0','algo':'${algo}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_sl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_rl_${algo}/model'}" >> ${rl4rs_output_dir}/eval_simulator_a_rl_sl_${algo}.log &&

python -u simulator_eval.py "{'gpu':False,'env':'SeqSlateRecEnv-v0','algo':'${algo}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_sl_${algo}/model'}" >> ${rl4rs_output_dir}/eval_simulator_b2_sl_rl_${algo}.log &&
python -u simulator_eval.py "{'gpu':False,'env':'SeqSlateRecEnv-v0','algo':'${algo}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_sl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_rl_${algo}/model'}" >> ${rl4rs_output_dir}/eval_simulator_b2_rl_sl_${algo}.log

echo '1'