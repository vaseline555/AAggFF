#!/bin/sh

## CelebA: 9,343 clients, 2 classes

python3 main.py \
--exp_name "[SWEEP] CelebA_AFL (lambda_lr=0.01)" --seed 1 --device cuda:1 \
--dataset CelebA \
--split_type pre --test_size 0.2 \
--model_name CelebACNN --hidden_size 32 --resize 84 --imnorm \
--algorithm afl --fair_const 0.01 --eval_fraction 1 --eval_type local --eval_every 3000 --eval_metrics acc1 \
--R 3000 --C 0.00054 --E 1 --B 20 \
--optimizer SGD --weight_decay 1e-4 --lr 0.1 --lr_decay 0.965 --lr_decay_step 300 --server_lr 1.0 --criterion BCEWithLogitsLoss &

python3 main.py \
--exp_name "[SWEEP] CelebA_AFL (lambda_lr=0.1)" --seed 1 --device cuda:1 \
--dataset CelebA \
--split_type pre --test_size 0.2 \
--model_name CelebACNN --hidden_size 32 --resize 84 --imnorm \
--algorithm afl --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 3000 --eval_metrics acc1 \
--R 3000 --C 0.00054 --E 1 --B 20 \
--optimizer SGD --weight_decay 1e-4 --lr 0.1 --lr_decay 0.965 --lr_decay_step 300 --server_lr 1.0 --criterion BCEWithLogitsLoss &

python3 main.py \
--exp_name "[SWEEP] CelebA_AFL (lambda_lr=1.0)" --seed 1 --device cuda:1 \
--dataset CelebA \
--split_type pre --test_size 0.2 \
--model_name CelebACNN --hidden_size 32 --resize 84 --imnorm \
--algorithm afl --fair_const 1.0 --eval_fraction 1 --eval_type local --eval_every 3000 --eval_metrics acc1 \
--R 3000 --C 0.00054 --E 1 --B 20 \
--optimizer SGD --weight_decay 1e-4 --lr 0.1 --lr_decay 0.965 --lr_decay_step 300 --server_lr 1.0 --criterion BCEWithLogitsLoss   
