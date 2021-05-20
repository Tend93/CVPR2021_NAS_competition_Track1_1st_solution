# CVPR2021_NAS_competition_Track1_1st_solution

This project is reproduction code for CVPR2021 NAS competition Track1 1st solution

## Reproduction Process ########################

#### Preparing 

1. get cifar100 dataset from https://aistudio.baidu.com/aistudio/projectdetail/1720490?channelType=0&channel=0

2. put cifar100 dataset into folder "data"

### Environment

Ubuntu 16.04.6

python 3.6.5

pytorch-1.4.0

CUDA Version 10.1.243

TITAN XP

### SPN train

#### step 1.train max-path ResNet20:

> ./script_train_net.sh

#### step 2.train supernet with KL loss distillation: 

> ./script_train_spn_dtkl_gs.sh

#### step 3.train multi-branch supernet with distillation: 

> ./script_train_spn_branch_multi.sh

#### step 4.multi-branch supernet training stage2: 

> ./script_train_spn_branch_multi_2stage.sh

### SPN test:
> ./script_run_batch_test.sh

### log and checkpoint files list

#### test results
./submit/spn0517_2stage_bma_prl48p1_100_1e3_cls02_p59 - test logs of results submitted to leader-board with best pearson score
./submit/spn0517_2stage_bmafix_prl48p1_100_1e3_cls0_p79 - test logs of results last submitted to leader-board

#### train results
./net/maxpath2spn-20210408-204246 - logs and checkpoints by SPN train step 1
./net/spn0513_dtkls8_lk5_prl48p1_cls02-20210514-170015 - logs and checkpoints by SPN train step 2
./net/spn0507_bmufix_s8_prl48p1_selfdt_1e3_300-20210515-230111 - logs and checkpoints by SPN train step 3
./net/spn0517_2stage_bma_prl48p1_100_1e3_cls02-20210517-174852 - logs and checkpoints by SPN train step 4
./net/spn0517_2stage_bmafix_prl48p1_100_1e3_cls0-20210517-174852 - logs and checkpoints by SPN train step 4
