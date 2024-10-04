#!/bin/sh

## Heart: 4 clients, 2 classes

for s in 2 4 6
do
    yes | python3 main.py \
    --exp_name "[PnP ($s)] Heart_FedAvg_AAggFF" --seed $s --device cuda:0 \
    --dataset Heart \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm aaggff --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
    --R 200 --E 1 --B 20 --C 1.0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

    yes | python3 main.py \
    --exp_name "[PnP ($s)] Heart_FedProx_AAggFF" --seed $s --device cuda:1 \
    --dataset Heart \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm aaggffprox --fair_const 2 --mu 0.001 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
    --R 200 --E 1 --B 20 --C 1.0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &
    
    yes | python3 main.py \
    --exp_name "[PnP ($s)] Heart_FedAdam_AAggFF" --seed $s --device cuda:2 \
    --dataset Heart \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm aaggffadam --fair_const 2 --beta1 0.9 --beta2 0.999 --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
    --R 200 --E 1 --B 20 --C 1.0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

    yes | python3 main.py \
    --exp_name "[PnP ($s)] Heart_FedYogi_AAggFF" --seed $s --device cuda:0 \
    --dataset Heart \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm aaggffyogi --fair_const 2 --beta1 0.9 --beta2 0.999 --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
    --R 200 --E 1 --B 20 --C 1.0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

    yes | python3 main.py \
    --exp_name "[PnP ($s)] Heart_FedAdagrad_AAggFF" --seed $s --device cuda:1 \
    --dataset Heart \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm aaggffadagrad --fair_const 2 --beta1 0. --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
    --R 200 --E 1 --B 20 --C 1.0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &
    
    yes | python3 main.py \
    --exp_name "[PnP ($s)] Heart_FedAvg" --seed $s --device cuda:2 \
    --dataset Heart \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
    --R 200 --E 1 --B 20 --C 1.0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

    yes | python3 main.py \
    --exp_name "[PnP ($s)] Heart_FedProx" --seed $s --device cuda:0 \
    --dataset Heart \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm fedprox --mu 0.001 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
    --R 200 --E 1 --B 20 --C 1.0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &
    
    yes | python3 main.py \
    --exp_name "[PnP ($s)] Heart_FedAdam" --seed $s --device cuda:1 \
    --dataset Heart \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm fedadam --beta1 0.9 --beta2 0.999 --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
    --R 200 --E 1 --B 20 --C 1.0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

    yes | python3 main.py \
    --exp_name "[PnP ($s)] Heart_FedYogi" --seed $s --device cuda:2 \
    --dataset Heart \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm fedyogi --beta1 0.9 --beta2 0.999 --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
    --R 200 --E 1 --B 20 --C 1.0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss &

    yes | python3 main.py \
    --exp_name "[PnP ($s)] Heart_FedAdagrad" --seed $s --device cuda:0 \
    --dataset Heart \
    --split_type pre --test_size 0.2 \
    --model_name LogReg --num_layers 1 \
    --algorithm fedadagrad --beta1 0. --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 20 --eval_metrics auroc \
    --R 200 --E 1 --B 20 --C 1.0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 2 --criterion BCEWithLogitsLoss
    wait
done

