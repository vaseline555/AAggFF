#!/bin/sh

## ISIC2019: 6 clients, 8 classes

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_AAggFF (0)" --seed 1 --device cuda:0 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 0 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 50 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_AAggFF (1)" --seed 1 --device cuda:1 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 1 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 50 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_AAggFF (2)" --seed 1 --device cuda:2 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 50 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_AAggFF (3)" --seed 1 --device cuda:0 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 50 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_AAggFF (4)" --seed 1 --device cuda:1 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 4 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 50 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &
 
yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_AAggFF (5)" --seed 1 --device cuda:2 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 5 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 50 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss
