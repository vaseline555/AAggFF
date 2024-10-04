#!/bin/sh

## ISIC2019: 6 clients, 8 classes

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_AFL (lambda_lr=0.01)" --seed 1 --device cuda:0 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm afl --fair_const 0.01 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 50 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_AFL (lambda_lr=0.1)" --seed 1 --device cuda:1 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm afl --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 50 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_AFL (lambda_lr=1.0)" --seed 1 --device cuda:2 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm afl --fair_const 1.0 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 50 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss 