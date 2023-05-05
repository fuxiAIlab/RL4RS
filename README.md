# RL4RS: A Real-World Dataset for Reinforcement Learning based Recommender System
<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) -->

[![License](https://licensebuttons.net/l/by/3.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

RL4RS is a real-world deep reinforcement learning recommender system dataset for practitioners and researchers.

```py
import gym
from rl4rs.env.slate import SlateRecEnv, SlateState

sim = SlateRecEnv(config, state_cls=SlateState)
env = gym.make('SlateRecEnv-v0', recsim=sim)
for i in range(epoch):
    obs = env.reset()
    for j in range(config["max_steps"]):
        action = env.offline_action
        next_obs, reward, done, info = env.step(action)
        if done[0]:
            break
```
Dataset Download(data only): https://zenodo.org/record/6622390#.YqBBpRNBxQK

Dataset Download(for reproduction): https://drive.google.com/file/d/1YbPtPyYrMvMGOuqD4oHvK0epDtEhEb9v/view?usp=sharing

Paper: https://arxiv.org/pdf/2110.11073.pdf

<!--Paper_latest: https://openreview.net/pdf?id=euli0I5CKvy-->

Appendix: https://github.com/fuxiAIlab/RL4RS/blob/main/RL4RS_appendix.pdf

Kaggle Competition (old version): https://www.kaggle.com/c/bigdata2021-rl-recsys/overview

Resource Page: https://fuxi-up-research.gitbook.io/fuxi-up-challenges/

## RL4RS News
![new](/assets/new.gif) **04/20/2023**: SIGIR 2023 Resource Track, [Accept].

**09/02/2022**: We release RL4RS [v1.1.0](https://github.com/fuxiAIlab/RL4RS/releases/tag/v1.1.0). 1) two additional RS datasets for comparison, Last.fm and CIKMCup2016; 2) two additional model-free baselines, TD3 and RAINBOW, and two additional model-based batch RL baselines, MOPO (Model-based Offline Policy Optimization) and COMBO(Conservative Offline Model-Based Policy Optimization). 3) BCQ and CQL support continuous action spaces. 

<!--**08/28/2022**: NeurIPS 2022 Track Datasets and Benchmarks, [Under Review](https://openreview.net/forum?id=euli0I5CKvy).-->

**09/17/2022**: A hand-on Invited talk at [DRL4IR Workshop](https://drl4ir.github.io/), SIGIR2022.

**12/17/2021**: Hosting [IEEE BigData2021 Cup Challenges](http://bigdataieee.org/BigData2021/BigDataCupChallenges.html), [Track I](https://www.kaggle.com/c/bigdata2021-rl-recsys/overview) for Supervised Learning and [Track II](https://fuxi-up-research.gitbook.io/fuxi-up-challenges/challenge/bigdatacup2021-rl4rs-challenge) for Reinforcement Learning.


## key features

### :star: Real-World Datasets
- **two real-world datasets**: Besides the artificial datasets or semi-simulated datasets, RL4RS collects the raw logged data from one of the most popular games released by NetEase Game, which is naturally a sequential decision-making problem.
- **data understanding tool**: RL4RS provides a data understanding tool for testing the proper use of RL on recommendation system datasets.
- **advanced dataset setting**: RL4RS provides the separated data before and after reinforcement learning deployment for each dataset, which can simulate the difficulties to train a good RL policy from the dataset collected by SL-based algorithm.

### :zap: Practical RL Baselines
- **model-free RL**: RL4RS supports state-of-the-art RL libraries, such as RLlib and Tianshou. We provide the example codes of state-of-the-art model-free algorithms (A2C, PPO, etc.) implemented by RLlib library on both discrete and continue (combining policy gradients with a K-NN search) RL4RS environment.
- **offline RL**: RL4RS implements offline RL algorithms including BC, BCQ and CQL through d3rlpy library. RL4RS is also the first to report the effectiveness of offline RL algorithms (BCQ and CQL) in RL-based RS domain.
- **RL-based RS baselines**: RL4RS implements some algorithms proposed in the RL-based RS domain, including Exact-k and Adversarial User Model.
- **offline RL evaluation**: In addition to the reward indicator and traditional RL evaluation setting (train and test on the same environment), RL4RS try to provide a complete evaluation framework by placing more emphasis on counterfactual policy evaluation.

### :beginner: Easy-To-Use scaleable API
- **low coupling structure**: RL4RS specifies a fixed data format to reduce code coupling. And the data-related logics are unified into data preprocessing scripts or user-defined state classes.
- **file-based RL environment**: RL4RS implements a file-based gym environment, which enables random sampling and sequential access to datasets exceeding memory size. It is easy to extend it to distributed file systems.
- **http-based vector Env**: RL4RS naturally supports Vector Env, that is, the environment processes batch data at one time. We further encapsulate the env through the HTTP interface, so that it can be deployed on multiple servers to accelerate the generation of samples.
       
## experimental features (welcome contributions!)
- A new dataset for bundle recommendation with variable discounts, flexible recommendation trigger, and modifiable item content is in prepare.
- Take raw feature rather than hidden layer embedding as observation input for offline RL
- Model-based RL Algorithms 
- Reward-oriented simulation environment construction
- reproduce more algorithms (RL models, safe exploration techniques, etc.) proposed in RL-based RS domain
- Support Parametric-Action DQN, in which we input concatenated state-action pairs and output the Q-value for each pair.

                                     

## installation
RL4RS supports Linux, at least 64 GB Mem !!

### Github (recommended)
```
$ git clone https://github.com/fuxiAIlab/RL4RS
$ export PYTHONPATH=$PYTHONPATH:`pwd`/rl4rs
$ conda env create -f environment.yml
$ conda activate rl4rs
```

### Dataset Download (Google Driver) 
Dataset Download: https://drive.google.com/file/d/1YbPtPyYrMvMGOuqD4oHvK0epDtEhEb9v/view?usp=sharing

```
.
|-- batchrl
|   |-- BCQ_SeqSlateRecEnv-v0_b_all.h5
|   |-- BCQ_SlateRecEnv-v0_a_all.h5
|   |-- BC_SeqSlateRecEnv-v0_b_all.h5
|   |-- BC_SlateRecEnv-v0_a_all.h5
|   |-- CQL_SeqSlateRecEnv-v0_b_all.h5
|   `-- CQL_SlateRecEnv-v0_a_all.h5
|-- data_understanding_tool
|   |-- dataset
|   |   |-- ml-25m.zip
|   |   `-- yoochoose-clicks.dat.zip
|   `-- finetuned
|       |-- movielens.csv
|       |-- movielens.h5
|       |-- recsys15.csv
|       |-- recsys15.h5
|       |-- rl4rs.csv
|       `-- rl4rs.h5
|-- exactk
|   |-- exact_k.ckpt.10000.data-00000-of-00001
|   |-- exact_k.ckpt.10000.index
|   `-- exact_k.ckpt.10000.meta
|-- ope
|   `-- logged_policy.h5
|-- raw_data
|   |-- item_info.csv
|   |-- rl4rs_dataset_a_rl.csv
|   |-- rl4rs_dataset_a_sl.csv
|   |-- rl4rs_dataset_b_rl.csv
|   `-- rl4rs_dataset_b_sl.csv
`-- simulator
    |-- finetuned
    |   |-- simulator_a_dien
    |   |   |-- checkpoint
    |   |   |-- model.data-00000-of-00001
    |   |   |-- model.index
    |   |   `-- model.meta
    |   `-- simulator_b2_dien
    |       |-- checkpoint
    |       |-- model.data-00000-of-00001
    |       |-- model.index
    |       `-- model.meta
    |-- rl4rs_dataset_a_shuf.csv
    `-- rl4rs_dataset_b3_shuf.csv
```

## two ways to use this resource
### Reinforcement Learning Only 
```
# move simulator/*.csv to rl4rs/dataset
# move simulator/finetuned/* to rl4rs/output
cd reproductions/
# run exact-k
bash run_exact_k.sh
# start http-based Env, then run RLlib library
nohup python -u rl4rs/server/gymHttpServer.py &
bash run_modelfree_rl.sh DQN/PPO/DDPG/PG/PG_conti/etc.
```

### start from scratch (batch-rl, environment simulation, etc.)
```
cd reproductions/
# first step, generate tfrecords for supervised learning (environment simulation) 
# is time-consuming, you can annotate them firstly.
bash run_split.sh

# environment simulation part (need tfrecord)
# run these scripts to compare different SL methods
bash run_supervised_item.sh dnn/widedeep/dien/lstm
bash run_supervised_slate.sh dnn_slate/adversarial_slate/etc.
# or you can directly train DIEN-based simulator as RL Env.
bash run_simulator_train.sh dien

# model-free part (need run_simulator_train.sh)
# run exact-k
bash run_exact_k.sh
# start http-based Env, then run RLlib library
nohup python -u rl4rs/server/gymHttpServer.py &
bash run_modelfree_rl.sh DQN/PPO/DDPG/PG/PG_conti/etc.

# offline RL part (need run_simulator_train.sh)
# generate offline dataset for offline RL first (dataset_generate stage)
# generate offline dataset for offline RL first (train stage)
bash run_batch_rl.sh BC/BCQ/CQL
```

## reported baselines 
| algorithm  | category | support mode |
|:-|:-:|:-:|
| [Wide&Deep](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454) | supervised learning | item-wise classification/slate-wise classification/item ranking |
| [GRU4Rec](https://arxiv.org/pdf/1511.06939) | supervised learning | item-wise classification/slate-wise classification/item ranking |
| [DIEN](https://www.researchgate.net/profile/Xiaoqiang-Zhu-7/publication/327591686_Deep_Interest_Evolution_Network_for_Click-Through_Rate_Prediction/links/5bc0398f458515a7a9e2a6db/Deep-Interest-Evolution-Network-for-Click-Through-Rate-Prediction.pdf) | supervised learning | item-wise classification/slate-wise classification/item ranking |
| [Adversarial User Model](http://proceedings.mlr.press/v97/chen19f/chen19f.pdf) | supervised learning | item-wise classification/slate-wise classification/item ranking |
| [Exact-K](https://arxiv.org/pdf/1905.07089.pdf) | model-free learning | discrete env & hidden state as observation |
| [Policy Gredient (PG)](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) | model-free RL | model-free learning | discrete/conti env & raw feature/hidden state as observation |
| [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236) | model-free RL | discrete env & raw feature/hidden state as observation |
| [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | model-free RL | conti env & raw feature/hidden state as observation |
| [Asynchronous Actor-Critic (A2C)](http://proceedings.mlr.press/v48/mniha16.pdf) | model-free RL | discrete/conti env & raw feature/hidden state as observation |
| [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) | model-free RL | discrete/conti env & raw feature/hidden state as observation |
| Behavior Cloning | supervised learning/Offline RL | discrete env & hidden state as observation |
| [Batch Constrained Q-learning (BCQ)](https://arxiv.org/abs/1812.02900) | Offline RL | discrete env & hidden state as observation |
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | Offline RL | discrete env & hidden state as observation |

## supported algorithms (from RLlib and d3rlpy)
| algorithm | discrete control | continuous control | offline RL? |
|:-|:-:|:-:|:-:|
| Behavior Cloning (supervised learning) | :white_check_mark: | :white_check_mark: | |
| [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236) | :white_check_mark: | :no_entry: | |
| [Double DQN](https://arxiv.org/abs/1509.06461) | :white_check_mark: | :no_entry: | |
| [Rainbow](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17204/16680) | :white_check_mark: | :no_entry: | |
| [PPO](https://arxiv.org/pdf/1707.06347.pdf) | :white_check_mark: | :white_check_mark: | |
| [A2C A3C](http://proceedings.mlr.press/v48/mniha16.pdf) | :white_check_mark: | :white_check_mark: | |
| [IMPALA](https://arxiv.org/pdf/1802.01561.pdf) | :white_check_mark: | :white_check_mark: | |
| [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | :no_entry: | :white_check_mark: | |
| [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/abs/1802.09477) | :no_entry: | :white_check_mark: | |
| [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905) | :white_check_mark: | :white_check_mark: | |
| [Batch Constrained Q-learning (BCQ)](https://arxiv.org/abs/1812.02900) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Bootstrapping Error Accumulation Reduction (BEAR)](https://arxiv.org/abs/1906.00949) | :no_entry: | :white_check_mark: | :white_check_mark: |
| [Advantage-Weighted Regression (AWR)](https://arxiv.org/abs/1910.00177) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Advantage Weighted Actor-Critic (AWAC)](https://arxiv.org/abs/2006.09359) | :no_entry: | :white_check_mark: | :white_check_mark: |
| [Critic Reguralized Regression (CRR)](https://arxiv.org/abs/2006.15134) | :no_entry: | :white_check_mark: | :white_check_mark: |
| [Policy in Latent Action Space (PLAS)](https://arxiv.org/abs/2011.07213) | :no_entry: | :white_check_mark: | :white_check_mark: |
| [TD3+BC](https://arxiv.org/abs/2106.06860) | :no_entry: | :white_check_mark: | :white_check_mark: |


## examples
See script/ and reproductions/.

RLlib examples: https://docs.ray.io/en/latest/rllib-examples.html

d3rlpy examples: https://d3rlpy.readthedocs.io/en/v1.0.0/

## reproductions
See reproductions/.
```bash
bash run_xx.sh ${param}
```
| experiment in the paper  | shell script | optional param. | description | 
|:-|:-:|:-:|:-:|
| Sec.3 | run_split.sh  | - | dataset split/shuffle/align(for datasetB)/to tfrecord |
| Sec.4 | run_mdp_checker.sh | recsys15/movielens/rl4rs | unzip ml-25m.zip and yoochoose-clicks.dat.zip into dataset/ |
| Sec.5.1 | run_supervised_item.sh | dnn/widedeep/lstm/dien | Table 5. Item-wise classification |
| Sec.5.1 | run_supervised_slate.sh | dnn_slate/widedeep_slate/lstm_slate/dien_slate/adversarial_slate | Table 5. Item-wise rank |
| Sec.5.1 | run_supervised_slate.sh | dnn_slate_multiclass/widedeep_slate_multiclass/lstm_slate_multiclass/dien_slate_multiclass | Table 5. Slate-wise classification |
| Sec.5.1 & Sec.6 | run_simulator_train.sh | dien | dien-based simulator for different trainsets |
| Sec.5.1 & Sec.6 | run_simulator_eval.sh | dien | Table 6. |
| Sec.5.1 & Sec.6 | run_modelfree_rl.sh | PG/DQN/A2C/PPO/IMPALA/DDPG/*_conti | Table 7. |
| Sec.5.2 & Sec.6 | run_batch_rl.sh | BC/BCQ/CQL | Table 8. |
| Sec.5.1 | run_exact_k.sh | - | Exact-k |
| - | run_simulator_env_test.sh | - | examining the consistency of features (observations) between RL env and supervised simulator |


## contributions
Any kind of contribution to RL4RS would be highly appreciated!
Please contact us by email.

## community
| Channel | Link |
|:-|:-|
| Materials | [Google Drive](https://drive.google.com/file/d/1YbPtPyYrMvMGOuqD4oHvK0epDtEhEb9v/view?usp=sharing) |
| Email | [Mail](asdqsczser@gmail.com) |
| Issues | [GitHub Issues](https://github.com/fuxiAIlab/RL4RS/issues) |
| Fuxi Team | [Fuxi HomePage](https://fuxi.163.com/en/) |
| Our Team | [Open-project](https://fuxi-up-research.gitbook.io/open-project/) |

## citation
```
@article{2021RL4RS,
title={RL4RS: A Real-World Benchmark for Reinforcement Learning based Recommender System},
author={ Kai Wang and Zhene Zou and Yue Shang and Qilin Deng and Minghao Zhao and Runze Wu and Xudong Shen and Tangjie Lyu and Changjie Fan},
journal={ArXiv},
year={2021},
volume={abs/2110.11073}
}
```


