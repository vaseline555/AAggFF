#!/bin/sh

## CelebA: 9,343 clients, 2 classes

for s in 2 4 6
do
    python3 main.py \
    --exp_name "[MAIN ($s)] CelebA_FedAvg" --seed $s --device cuda:0 \
    --dataset CelebA \
    --split_type pre --test_size 0.2 \
    --model_name CelebACNN --hidden_size 32 --resize 84 --imnorm \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 3000 --eval_metrics acc1 \
    --R 3000 --C 0.00054 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-4 --lr 0.1 --lr_decay 0.96 --lr_decay_step 300 --server_lr 1.0 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] CelebA_FedMGDA" --seed $s --device cuda:1 \
    --dataset CelebA \
    --split_type pre --test_size 0.2 \
    --model_name CelebACNN --hidden_size 32 --resize 84 --imnorm \
    --algorithm fedmgda --fair_const 0.5 --eval_fraction 1 --eval_type local --eval_every 3000 --eval_metrics acc1 \
    --R 3000 --C 0.00054 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-4 --lr 0.1 --lr_decay 0.96 --lr_decay_step 300 --server_lr 1.0 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] CelebA_PropFair" --seed $s --device cuda:2 \
    --dataset CelebA \
    --split_type pre --test_size 0.2 \
    --model_name CelebACNN --hidden_size 32 --resize 84 --imnorm \
    --algorithm propfair --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 3000 --eval_metrics acc1 \
    --R 3000 --C 0.00054 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-4 --lr 0.1 --lr_decay 0.96 --lr_decay_step 300 --server_lr 1.0 --criterion BCEWithLogitsLoss &&

    python3 main.py \
    --exp_name "[MAIN ($s)] CelebA_qFedAvg" --seed $s --device cuda:0 \
    --dataset CelebA \
    --split_type pre --test_size 0.2 \
    --model_name CelebACNN --hidden_size 32 --resize 84 --imnorm \
    --algorithm qfedavg --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 3000 --eval_metrics acc1 \
    --R 3000 --C 0.00054 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-4 --lr 0.1 --lr_decay 0.96 --lr_decay_step 300 --server_lr 1.0 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] CelebA_TERM" --seed $s --device cuda:1 \
    --dataset CelebA \
    --split_type pre --test_size 0.2 \
    --model_name CelebACNN --hidden_size 32 --resize 84 --imnorm \
    --algorithm term --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 3000 --eval_metrics acc1 \
    --R 3000 --C 0.00054 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-4 --lr 0.1 --lr_decay 0.96 --lr_decay_step 300 --server_lr 1.0 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] CelebA_AFL" --seed $s --device cuda:2 \
    --dataset CelebA \
    --split_type pre --test_size 0.2 \
    --model_name CelebACNN --hidden_size 32 --resize 84 --imnorm \
    --algorithm afl --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 3000 --eval_metrics acc1 \
    --R 3000 --C 0.00054 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-4 --lr 0.1 --lr_decay 0.96 --lr_decay_step 300 --server_lr 1.0 --criterion BCEWithLogitsLoss &&

    python3 main.py \
    --exp_name "[MAIN ($s)] CelebA_AAggFF" --seed $s --device cuda:0 \
    --dataset CelebA \
    --split_type pre --test_size 0.2 \
    --model_name CelebACNN --hidden_size 32 --resize 84 --imnorm \
    --algorithm aaggff --fair_const 4 --eval_fraction 1 --eval_type local --eval_every 3000 --eval_metrics acc1 \
    --R 3000 --C 0.00054 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-4 --lr 0.1 --lr_decay 0.96 --lr_decay_step 300 --server_lr 1.0 --criterion BCEWithLogitsLoss &
    wait
done