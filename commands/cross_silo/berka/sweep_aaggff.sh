#!/bin/sh

## Berka: 7 clients, 2 classes

python3 main.py \
--exp_name "[SWEEP] Berka_AAggFF (0)" --seed 1 --device cuda:1 \
--dataset Berka \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 0 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss &&

python3 main.py \
--exp_name "[SWEEP] Berka_AAggFF (1)" --seed 1 --device cuda:1 \
--dataset Berka \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 1 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss &&

python3 main.py \
--exp_name "[SWEEP] Berka_AAggFF (2)" --seed 1 --device cuda:1 \
--dataset Berka \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss &&

python3 main.py \
--exp_name "[SWEEP] Berka_AAggFF (3)" --seed 1 --device cuda:1 \
--dataset Berka \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss &&

python3 main.py \
--exp_name "[SWEEP] Berka_AAggFF (4)" --seed 1 --device cuda:1 \
--dataset Berka \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 4 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss &&
 
python3 main.py \
--exp_name "[SWEEP] Berka_AAggFF (5)" --seed 1 --device cuda:1 \
--dataset Berka \
--split_type pre --test_size 0.2 \
--model_name LogReg --num_layers 1 \
--algorithm aaggff --fair_const 5 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss
