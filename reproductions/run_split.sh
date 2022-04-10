#!/bin/bash

conda activate rl4rs
script_abs=$(readlink -f "$0")
rl4rs_benchmark_dir=$(dirname $script_abs)/..
rl4rs_dataset_dir=${rl4rs_benchmark_dir}/dataset
script_dir=${rl4rs_benchmark_dir}/script
rl4rs_output_dir=${rl4rs_benchmark_dir}/output
mkdir $rl4rs_output_dir
export rl4rs_benchmark_dir && export rl4rs_output_dir && export rl4rs_dataset_dir

cd $rl4rs_dataset_dir

#raw dataset
awk -F "@" 'NR>1 {print}' rl4rs_dataset_a_sl.csv > rl4rs_dataset_a.csv &&
awk -F "@" 'NR>1 {print}' rl4rs_dataset_a_rl.csv >> rl4rs_dataset_a.csv &&
awk -F "@" 'NR>1 {print}' rl4rs_dataset_b_sl.csv > rl4rs_dataset_b.csv &&
awk -F "@" 'NR>1 {print}' rl4rs_dataset_b_rl.csv >> rl4rs_dataset_b.csv &&

#train/test split
awk -F "@" 'NR>1 && $2%10<=5 {print}' rl4rs_dataset_a_sl.csv > rl4rs_dataset_a_sl_train.csv &&
awk -F "@" 'NR>1 && $2%10>=6 {print}' rl4rs_dataset_a_sl.csv > rl4rs_dataset_a_sl_test.csv &&
awk -F "@" 'NR>1 && $2%10<=5 {print}' rl4rs_dataset_a_rl.csv > rl4rs_dataset_a_rl_train.csv &&
awk -F "@" 'NR>1 && $2%10>=6 {print}' rl4rs_dataset_a_rl.csv > rl4rs_dataset_a_rl_test.csv &&

awk -F "@" 'NR>1 && $2%10<=5 {print}' rl4rs_dataset_b_sl.csv > rl4rs_dataset_b_sl_train.csv &&
awk -F "@" 'NR>1 && $2%10>=6 {print}' rl4rs_dataset_b_sl.csv > rl4rs_dataset_b_sl_test.csv &&
awk -F "@" 'NR>1 && $2%10<=5 {print}' rl4rs_dataset_b_rl.csv > rl4rs_dataset_b_rl_train.csv &&
awk -F "@" 'NR>1 && $2%10>=6 {print}' rl4rs_dataset_b_rl.csv > rl4rs_dataset_b_rl_test.csv &&

cat rl4rs_dataset_a_sl_train.csv >  rl4rs_dataset_a_train.csv &&
cat rl4rs_dataset_a_rl_train.csv >>  rl4rs_dataset_a_train.csv &&
cat rl4rs_dataset_b_sl_train.csv >  rl4rs_dataset_b_train.csv &&
cat rl4rs_dataset_b_rl_train.csv >>  rl4rs_dataset_b_train.csv &&

cat rl4rs_dataset_a_sl_test.csv >  rl4rs_dataset_a_test.csv &&
cat rl4rs_dataset_a_rl_test.csv >>  rl4rs_dataset_a_test.csv &&
cat rl4rs_dataset_b_sl_test.csv >  rl4rs_dataset_b_test.csv &&
cat rl4rs_dataset_b_rl_test.csv >>  rl4rs_dataset_b_test.csv &&

cat rl4rs_dataset_a_train.csv > rl4rs_dataset_a.csv &&
cat rl4rs_dataset_a_test.csv >> rl4rs_dataset_a.csv &&
cat rl4rs_dataset_b_train.csv > rl4rs_dataset_b.csv &&
cat rl4rs_dataset_b_test.csv >> rl4rs_dataset_b.csv  &&

#dataset_b
cd ${script_dir}  &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b_sl.csv" "${rl4rs_dataset_dir}/rl4rs_dataset_b2_sl.csv" "data_augment" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b_rl.csv" "${rl4rs_dataset_dir}/rl4rs_dataset_b2_rl.csv" "data_augment" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b_train.csv" "${rl4rs_dataset_dir}/rl4rs_dataset_b2_train.csv" "data_augment" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b_test.csv" "${rl4rs_dataset_dir}/rl4rs_dataset_b2_test.csv" "data_augment" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b.csv" "${rl4rs_dataset_dir}/rl4rs_dataset_b2.csv" "data_augment" &&

python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b2_sl.csv" "${rl4rs_dataset_dir}/rl4rs_dataset_b3_sl.csv" "slate2trajectory" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b2_rl.csv" "${rl4rs_dataset_dir}/rl4rs_dataset_b3_rl.csv" "slate2trajectory" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b2_train.csv" "${rl4rs_dataset_dir}/rl4rs_dataset_b3_train.csv" "slate2trajectory" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b2_test.csv" "${rl4rs_dataset_dir}/rl4rs_dataset_b3_test.csv" "slate2trajectory" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b2.csv" "${rl4rs_dataset_dir}/rl4rs_dataset_b3.csv" "slate2trajectory" &&


#shuffle for RL Env.
cd $rl4rs_dataset_dir &&
cat rl4rs_dataset_a.csv|shuf > rl4rs_dataset_a_shuf.csv &&
awk -F "@" 'NR>1 {print}' rl4rs_dataset_a_sl.csv|shuf > rl4rs_dataset_a_sl_shuf.csv &&
awk -F "@" 'NR>1 {print}' rl4rs_dataset_a_rl.csv|shuf > rl4rs_dataset_a_rl_shuf.csv &&
cat rl4rs_dataset_a_train.csv|shuf > rl4rs_dataset_a_train_shuf.csv &&
cat rl4rs_dataset_a_test.csv|shuf > rl4rs_dataset_a_test_shuf.csv &&
cat rl4rs_dataset_b3.csv|shuf > rl4rs_dataset_b3_shuf.csv &&
cat rl4rs_dataset_b3_sl.csv|shuf > rl4rs_dataset_b3_sl_shuf.csv &&
cat rl4rs_dataset_b3_rl.csv|shuf > rl4rs_dataset_b3_rl_shuf.csv &&
cat rl4rs_dataset_b3_train.csv|shuf > rl4rs_dataset_b3_train_shuf.csv &&
cat rl4rs_dataset_b3_test.csv|shuf > rl4rs_dataset_b3_test_shuf.csv &&


cd $(dirname $script_abs) &&
bash file_split.sh "rl4rs_dataset_a_sl_shuf.csv" &&
bash file_split.sh "rl4rs_dataset_a_rl_shuf.csv" &&
bash file_split.sh "rl4rs_dataset_a_train_shuf.csv" &&
bash file_split.sh "rl4rs_dataset_a_test_shuf.csv" &&
bash file_split.sh "rl4rs_dataset_a_shuf.csv" &&
bash file_split.sh "rl4rs_dataset_b2_sl.csv" &&
bash file_split.sh "rl4rs_dataset_b2_rl.csv" &&
bash file_split.sh "rl4rs_dataset_b2_train.csv" &&
bash file_split.sh "rl4rs_dataset_b2_test.csv" &&
bash file_split.sh "rl4rs_dataset_b2.csv"


#tfrecord for supervised learning
cd ${script_dir}

for ((i=0;i<5;i=i+1))
do
python data_preprocess.py "${rl4rs_output_dir}/rl4rs_dataset_a_sl_shuf.csv_000${i}.csv" "${rl4rs_output_dir}/rl4rs_dataset_a_sl.tfrecord.${i}" "tfrecord_item" &&
python data_preprocess.py "${rl4rs_output_dir}/rl4rs_dataset_a_rl_shuf.csv_000${i}.csv" "${rl4rs_output_dir}/rl4rs_dataset_a_rl.tfrecord.${i}" "tfrecord_item" &&
python data_preprocess.py "${rl4rs_output_dir}/rl4rs_dataset_a_train_shuf.csv_000${i}.csv" "${rl4rs_output_dir}/rl4rs_dataset_a_train.tfrecord.${i}" "tfrecord_item" &&
python data_preprocess.py "${rl4rs_output_dir}/rl4rs_dataset_a_test_shuf.csv_000${i}.csv" "${rl4rs_output_dir}/rl4rs_dataset_a_test.tfrecord.${i}" "tfrecord_item" &&
python data_preprocess.py "${rl4rs_output_dir}/rl4rs_dataset_a_shuf.csv_000${i}.csv" "${rl4rs_output_dir}/rl4rs_dataset_a.tfrecord.${i}" "tfrecord_item" &&
python data_preprocess.py "${rl4rs_output_dir}/rl4rs_dataset_b2_sl.csv_000${i}.csv" "${rl4rs_output_dir}/rl4rs_dataset_b2_sl.tfrecord.${i}" "tfrecord_item" &&
python data_preprocess.py "${rl4rs_output_dir}/rl4rs_dataset_b2_rl.csv_000${i}.csv" "${rl4rs_output_dir}/rl4rs_dataset_b2_rl.tfrecord.${i}" "tfrecord_item" &&
python data_preprocess.py "${rl4rs_output_dir}/rl4rs_dataset_b2_train.csv_000${i}.csv" "${rl4rs_output_dir}/rl4rs_dataset_b2_train.tfrecord.${i}" "tfrecord_item" &&
python data_preprocess.py "${rl4rs_output_dir}/rl4rs_dataset_b2_test.csv_000${i}.csv" "${rl4rs_output_dir}/rl4rs_dataset_b2_test.tfrecord.${i}" "tfrecord_item" &&
python data_preprocess.py "${rl4rs_output_dir}/rl4rs_dataset_b2.csv_000${i}.csv" "${rl4rs_output_dir}/rl4rs_dataset_b2.tfrecord.${i}" "tfrecord_item" &&
echo "1"
done

cd ${script_dir} &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_a_train_shuf.csv" "${rl4rs_output_dir}/rl4rs_dataset_a_train_slate.tfrecord" "tfrecord_slate" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_a_test_shuf.csv" "${rl4rs_output_dir}/rl4rs_dataset_a_test_slate.tfrecord" "tfrecord_slate" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b2_train.csv" "${rl4rs_output_dir}/rl4rs_dataset_b2_train_slate.tfrecord" "tfrecord_slate" &&
python data_preprocess.py "${rl4rs_dataset_dir}/rl4rs_dataset_b2_test.csv" "${rl4rs_output_dir}/rl4rs_dataset_b2_test_slate.tfrecord" "tfrecord_slate" &&

echo "1"
