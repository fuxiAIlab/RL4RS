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

# experiment on a_all env (model_file), train on a_all sample and test on a_all sample
python -u batchrl_train.py $algo 'dataset_generate' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_all'}" &&
#python -u batchrl_train.py $algo 'train' "{'env':'SlateRecEnv-v0','trial_name':'a_all'}" >> ${rl4rs_output_dir}/batchrl_a_all_${algo}.log &&
#python -u batchrl_train.py $algo 'eval' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_all','gpu':False}" >> ${rl4rs_output_dir}/batchrl_a_all_${algo}.log &&
#python -u batchrl_train.py $algo 'ope' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_all','gpu':False}" >> ${rl4rs_output_dir}/batchrl_a_all_${algo}.log &&


# experiment on a_all env (model_file), train on a_train sample and test on a_test sample
python -u batchrl_train.py $algo 'dataset_generate' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_train_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_train'}" &&
#python -u batchrl_train.py $algo 'train' "{'env':'SlateRecEnv-v0','trial_name':'a_train'}" >> ${rl4rs_output_dir}/batchrl_a_train_${algo}.log &&
#python -u batchrl_train.py $algo 'eval' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_test_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_train','gpu':False}" >> ${rl4rs_output_dir}/batchrl_a_train_${algo}.log  &&
#python -u batchrl_train.py $algo 'ope' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_test_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_train','gpu':False}" >> ${rl4rs_output_dir}/batchrl_a_train_${algo}.log  &&


# experiment train on a_sl env and test on a_rl sample
python -u batchrl_train.py $algo 'dataset_generate' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_sl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_sl_dien/model','trial_name':'a_sl'}" &&
#python -u batchrl_train.py $algo 'train' "{'env':'SlateRecEnv-v0','trial_name':'a_sl'}" >> ${rl4rs_output_dir}/batchrl_a_sl_${algo}.log  &&
#python -u batchrl_train.py $algo 'eval' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_sl_dien/model','trial_name':'a_sl','gpu':False}" >> ${rl4rs_output_dir}/batchrl_a_sl_${algo}.log &&
#python -u batchrl_train.py $algo 'ope' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_sl_dien/model','trial_name':'a_sl','gpu':False}" >> ${rl4rs_output_dir}/batchrl_a_sl_${algo}.log &&


# experiments on Sec 6.2
# experiment on a_all env, train on a_all env and test on a_rl sample
#python -u batchrl_train.py $algo 'eval' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_all','gpu':False}" >> ${rl4rs_output_dir}/eval_batchrl_a_all_all_rl_${algo}.log &&
# experiment on a_all env, train on a_sl env and test on a_rl sample
#python -u batchrl_train.py $algo 'dataset_generate' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_sl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_all_sl'}" &&
#python -u batchrl_train.py $algo 'train' "{'env':'SlateRecEnv-v0','trial_name':'a_all_sl'}" >> ${rl4rs_output_dir}/batchrl_a_all_sl_${algo}.log  &&
#python -u batchrl_train.py $algo 'eval' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_dien/model','trial_name':'a_all_sl','gpu':False}" >> ${rl4rs_output_dir}/eval_batchrl_a_all_sl_rl_${algo}.log &&
# experiment on a_sl env, train on a_sl env and test on a_rl sample
#python -u batchrl_train.py $algo 'eval' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_sl_dien/model','trial_name':'a_sl','gpu':False}" >> ${rl4rs_output_dir}/eval_batchrl_a_sl_sl_rl_${algo}.log &&
# experiment on a_sl env, train on a_sl env and test on env built from a_rl sample
#python -u batchrl_train.py $algo 'eval' "{'env':'SlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_a_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_a_rl_dien/model','trial_name':'a_sl','gpu':False}" >> ${rl4rs_output_dir}/eval_batchrl_a_rl_sl_rl_${algo}.log &&


# experiment on b_all env (model_file), train on b_all sample and test on b_all sample
python -u batchrl_train.py $algo 'dataset_generate' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_all'}" &&
#python -u batchrl_train.py $algo 'train' "{'env':'SeqSlateRecEnv-v0','trial_name':'b_all'}" >> ${rl4rs_output_dir}/batchrl_b_all_${algo}.log &&
#python -u batchrl_train.py $algo 'eval' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_all','gpu':False}" >> ${rl4rs_output_dir}/batchrl_b_all_${algo}.log &&
#python -u batchrl_train.py $algo 'ope' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_all','gpu':False}" >> ${rl4rs_output_dir}/batchrl_b_all_${algo}.log &&


# experiment on b_all env (model_file), train on b_train sample and test on b_b_test sample
python -u batchrl_train.py $algo 'dataset_generate' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_train_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_train'}" &&
#python -u batchrl_train.py $algo 'train' "{'env':'SeqSlateRecEnv-v0','trial_name':'b_train'}" >> ${rl4rs_output_dir}/batchrl_b_train_${algo}.log  &&
#python -u batchrl_train.py $algo 'eval' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_test_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_train','gpu':False}" >> ${rl4rs_output_dir}/batchrl_b_train_${algo}.log &&
#python -u batchrl_train.py $algo 'ope' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_test_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_train','gpu':False}" >> ${rl4rs_output_dir}/batchrl_b_train_${algo}.log &&


# experiment train on b_sl env and test on b_rl env
python -u batchrl_train.py $algo 'dataset_generate' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_sl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_sl_dien/model','trial_name':'b_sl'}" &&
#python -u batchrl_train.py $algo 'train' "{'env':'SeqSlateRecEnv-v0','trial_name':'b_sl'}" >> ${rl4rs_output_dir}/batchrl_b_sl_${algo}.log &&
#python -u batchrl_train.py $algo 'eval' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_sl_dien/model','trial_name':'b_sl','gpu':False}" >> ${rl4rs_output_dir}/batchrl_b_sl_${algo}.log &&
#python -u batchrl_train.py $algo 'ope' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_sl_dien/model','trial_name':'b_sl','gpu':False}" >> ${rl4rs_output_dir}/batchrl_b_sl_${algo}.log &&

# experiments on Sec 6.2
# experiment on b_all env, train on b_all env and test on b_rl sample
#python -u batchrl_train.py $algo 'eval' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_all','gpu':False}" >> ${rl4rs_output_dir}/eval_batchrl_b_all_all_rl_${algo}.log &&
# experiment on b_all env, train on b_sl env and test on b_rl sample
#python -u batchrl_train.py $algo 'dataset_generate' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_sl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_all_sl'}" &&
#python -u batchrl_train.py $algo 'train' "{'env':'SeqSlateRecEnv-v0','trial_name':'b_all_sl'}" >> ${rl4rs_output_dir}/batchrl_b_all_sl_${algo}.log  &&
#python -u batchrl_train.py $algo 'eval' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_dien/model','trial_name':'b_all_sl','gpu':False}" >> ${rl4rs_output_dir}/eval_batchrl_b_all_sl_rl_${algo}.log &&
# experiment on b_sl env, train on b_sl env and test on b_rl sample
#python -u batchrl_train.py $algo 'eval' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_sl_dien/model','trial_name':'b_sl','gpu':False}" >> ${rl4rs_output_dir}/eval_batchrl_b_sl_sl_rl_${algo}.log &&
# experiment on b_sl env, train on b_sl env and test on env built from b_rl sample
#python -u batchrl_train.py $algo 'eval' "{'env':'SeqSlateRecEnv-v0','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl_shuf.csv','model_file':'${rl4rs_output_dir}/simulator_b2_rl_dien/model','trial_name':'b_sl','gpu':False}" >> ${rl4rs_output_dir}/eval_batchrl_b_rl_sl_rl_${algo}.log &&


echo '1'