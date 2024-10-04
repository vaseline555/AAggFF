#!/bin/sh

## Heart: 4 clients, 2 classes

yes | python3 main.py \
--exp_name "[SWEEP] Heart_AAggFF (0)" --seed 1 --device cuda:0 \
--dataset Heart \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 0 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
--R 200 --E 1 --B 20 --C 1.0 \
--optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

yes | python3 main.py \
--exp_name "[SWEEP] Heart_AAggFF (1)" --seed 1 --device cuda:1 \
--dataset Heart \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 1 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
--R 200 --E 1 --B 20 --C 1.0 \
--optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

yes | python3 main.py \
--exp_name "[SWEEP] Heart_AAggFF (2)" --seed 1 --device cuda:2 \
--dataset Heart \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
--R 200 --E 1 --B 20 --C 1.0 \
--optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

yes | python3 main.py \
--exp_name "[SWEEP] Heart_AAggFF (3)" --seed 1 --device cuda:0 \
--dataset Heart \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
--R 200 --E 1 --B 20 --C 1.0 \
--optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

yes | python3 main.py \
--exp_name "[SWEEP] Heart_AAggFF (4)" --seed 1 --device cuda:1 \
--dataset Heart \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 4 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
--R 200 --E 1 --B 20 --C 1.0 \
--optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

yes | python3 main.py \
--exp_name "[SWEEP] Heart_AAggFF (5)" --seed 1 --device cuda:2 \
--dataset Heart \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 5 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
--R 200 --E 1 --B 20 --C 1.0 \
--optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss