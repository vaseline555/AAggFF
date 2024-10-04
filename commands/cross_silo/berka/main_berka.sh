#!/bin/sh

## Berka: 7 clients, 2 classes

for s in 2 4 6
do
    python3 main.py \
    --exp_name "[MAIN ($s)] Berka_FedAvg" --seed $s --device cuda:0 \
    --dataset Berka \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] Berka_AFL" --seed $s --device cuda:1 \
    --dataset Berka \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm afl --fair_const 0.01 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] Berka_qFedAvg" --seed $s --device cuda:2 \
    --dataset Berka \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm qfedavg --fair_const 1.0 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss 

    python3 main.py \
    --exp_name "[MAIN ($s)] Berka_TERM" --seed $s --device cuda:0 \
    --dataset Berka \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm term --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] Berka_PropFair" --seed $s --device cuda:1 \
    --dataset Berka \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm propfair --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] Berka_FedMGDA" --seed $s --device cuda:2 \
    --dataset Berka \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm fedmgda --fair_const 1.0 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] Berka_AAggFF" --seed $s --device cuda:0 \
    --dataset Berka \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm aaggff --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 1.0 --weight_decay 0.001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss
done
