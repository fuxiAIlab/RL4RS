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

# experiment in a_all env, train in a_all sample and test in a_all sample
python -u modelfree_train.py $algo "train" "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_all','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_a_all_${algo}.log &&
python -u modelfree_train.py $algo "eval" "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_all','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_a_all_${algo}.log &&
#python -u modelfree_train.py $algo "ope" "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_all','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_a_all_${algo}.log &&


# experiment in a_all env, train in a_train sample and test in a_test sample
#python -u modelfree_train.py $algo "train" "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_train_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_train','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_a_train_${algo}.log &&
#python -u modelfree_train.py $algo "eval" "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_test_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_train','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_a_train_${algo}.log &&
#python -u modelfree_train.py $algo "ope" "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_test_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_train','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_a_train_${algo}.log &&


# experiment train in a_sl env and test in a_rl env
#python -u modelfree_train.py $algo "train" "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_sl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_sl_dien/model','trial_name':'a_sl','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_a_sl_${algo}.log &&
#python -u modelfree_train.py $algo "eval" "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_rl_dien/model','trial_name':'a_sl','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_a_sl_${algo}.log &&
#python -u modelfree_train.py $algo "ope" "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_rl_dien/model','trial_name':'a_sl','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_a_sl_${algo}.log &&


# experiment in b_all env, train in b_all sample and test in b_all sample
python -u modelfree_train.py $algo "train" "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_all','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_b_all_${algo}.log &&
python -u modelfree_train.py $algo "eval" "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_all','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_b_all_${algo}.log &&
#python -u modelfree_train.py $algo "ope" "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_all','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_b_all_${algo}.log &&


# experiment in b_all env, train in b_train sample and test in b_test sample
#python -u modelfree_train.py $algo "train" "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_train_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_train','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_b_train_${algo}.log &&
#python -u modelfree_train.py $algo "eval" "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_test_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_train','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_b_train_${algo}.log &&
#python -u modelfree_train.py $algo "ope" "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_test_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_train','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_b_train_${algo}.log &&


# experiment train in b_sl env and test in b_rl env
#python -u modelfree_train.py $algo "train" "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_sl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_sl_dien/model','trial_name':'b_sl','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_b_sl_${algo}.log &&
#python -u modelfree_train.py $algo "eval" "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_rl_dien/model','trial_name':'b_sl','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_b_sl_${algo}.log &&
#python -u modelfree_train.py $algo "ope" "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_rl_dien/model','trial_name':'b_sl','remote_base':'http://127.0.0.1:5000'}" >> ${rl4rs_output_dir}/modelfree_b_sl_${algo}.log &&

echo "1"
