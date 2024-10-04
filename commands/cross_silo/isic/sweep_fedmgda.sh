#!/bin/sh

## ISIC2019: 6 clients, 8 classes

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_FedMGDA (eps=0.1)" --seed 1 --device cuda:0 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm fedmgda --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0000316 --weight_decay 0.00001 --lr_decay 0.99 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_FedMGDA (eps=0.5)" --seed 1 --device cuda:1 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm fedmgda --fair_const 0.5 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0000316 --weight_decay 0.00001 --lr_decay 0.99 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

yes | python3 main.py \
--exp_name "[SWEEP] ISIC2019_FedMGDA (eps=1.0)" --seed 1 --device cuda:2 \
--dataset ISIC2019 \
--split_type pre --test_size 0.2 \
--model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
--algorithm fedmgda --fair_const 1.0 --eval_fraction 1 --eval_type local --eval_every 5 --eval_metrics balacc acc1 acc5 \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.0000316 --weight_decay 0.00001 --lr_decay 0.99 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss 