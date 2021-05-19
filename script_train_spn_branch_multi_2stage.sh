#!/usr/bin/env bash
python train_spn_branch_multi_2stage.py --batch_size=128 --epochs=100 --gpu=0 --data='cifar100' --save=spn0517_2stage_bmafix_prl48p1_100_1e3_cls0 --root_path='./results/spn' --workers=32 --learning_rate=0.001 --lr_adjust_type='cosine' --weight_decay=5e-4 --momentum=0.9 --source_model='./results/spn/spn0507_bmufix_s8_prl48p1_selfdt_1e3_300-20210515-230111/checkpoint_ep299.pt' --sample_num=8 --fix_pretrain=True --dif_lr=False --distill_coeff=1.0
# python train_spn_branch_multi_2stage.py --batch_size=128 --epochs=100 --gpu=0 --data='cifar100' --save=spn0517_2stage_bma_prl48p1_100_1e3_cls02 --root_path='./results/spn' --workers=32 --learning_rate=0.001 --lr_adjust_type='cosine' --weight_decay=5e-4 --momentum=0.9 --source_model='./results/spn/spn0507_bmufix_s8_prl48p1_selfdt_1e3_300-20210515-230111/checkpoint_ep299.pt' --sample_num=8 --fix_pretrain=False --dif_lr=False --distill_coeff=0.8