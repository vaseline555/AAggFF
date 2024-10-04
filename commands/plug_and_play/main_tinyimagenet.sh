#!/bin/sh

## TinyImageNet: 1,000 clients, 200 classes

for s in 2 4 6
do
    python3 main.py \
    --exp_name "[PnP ($s)] TinyImageNet_FedAvg_AAggFF" --seed $s --device cuda:0 \
    --dataset TinyImageNet \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name MobileViT --resize 64 \
    --algorithm aaggff --fair_const 1 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
    --R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
    --optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[PnP ($s)] TinyImageNet_FedProx_AAggFF" --seed $s --device cuda:1 \
    --dataset TinyImageNet \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name MobileViT --resize 64 \
    --algorithm aaggffprox --fair_const 1 --mu 0.001 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
    --R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
    --optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &
    
    python3 main.py \
    --exp_name "[PnP ($s)] TinyImageNet_FedAdam_AAggFF" --seed $s --device cuda:2 \
    --dataset TinyImageNet \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name MobileViT --resize 64 \
    --algorithm aaggffadam --fair_const 1 --beta1 0.9 --beta2 0.999 --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
    --R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
    --optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[PnP ($s)] TinyImageNet_FedYogi_AAggFF" --seed $s --device cuda:0 \
    --dataset TinyImageNet \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name MobileViT --resize 64 \
    --algorithm aaggffyogi --fair_const 1 --beta1 0.9 --beta2 0.999 --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
    --R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
    --optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[PnP ($s)] TinyImageNet_FedAdagrad_AAggFF" --seed $s --device cuda:1 \
    --dataset TinyImageNet \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name MobileViT --resize 64 \
    --algorithm aaggffadagrad --fair_const 1 --beta1 0. --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
    --R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
    --optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &
    
    python3 main.py \
    --exp_name "[PnP ($s)] TinyImageNet_FedAvg" --seed $s --device cuda:2 \
    --dataset TinyImageNet \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name MobileViT --resize 64 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
    --R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
    --optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[PnP ($s)] TinyImageNet_FedProx" --seed $s --device cuda:0 \
    --dataset TinyImageNet \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name MobileViT --resize 64 \
    --algorithm fedprox --mu 0.001 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
    --R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
    --optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &
    
    python3 main.py \
    --exp_name "[PnP ($s)] TinyImageNet_FedAdam" --seed $s --device cuda:1 \
    --dataset TinyImageNet \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name MobileViT --resize 64 \
    --algorithm fedadam --beta1 0.9 --beta2 0.999 --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
    --R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
    --optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[PnP ($s)] TinyImageNet_FedYogi" --seed $s --device cuda:2 \
    --dataset TinyImageNet \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name MobileViT --resize 64 \
    --algorithm fedyogi --beta1 0.9 --beta2 0.999 --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
    --R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
    --optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[PnP ($s)] TinyImageNet_FedAdagrad" --seed $s --device cuda:0 \
    --dataset TinyImageNet \
    --split_type diri --cncntrtn 0.01 --test_size 0.2 \
    --model_name MobileViT --resize 64 \
    --algorithm fedadagrad --beta1 0. --tau 1e-4 --server_lr 1e-2 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
    --R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
    --optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss
    wait
done

