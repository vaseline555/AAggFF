#!/bin/sh

## TinyImageNet: 3,597 clients, 200 classes

python3 main.py \
--exp_name "[SWEEP] TinyImageNet_AAggFF (0)" --seed 1 --device cuda:0 \
--dataset TinyImageNet \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name MobileViT --resize 64 \
--algorithm aaggff --fair_const 0 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
--R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
--optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

python3 main.py \
--exp_name "[SWEEP] TinyImageNet_AAggFF (1)" --seed 1 --device cuda:1 \
--dataset TinyImageNet \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name MobileViT --resize 64 \
--algorithm aaggff --fair_const 1 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
--R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
--optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

python3 main.py \
--exp_name "[SWEEP] TinyImageNet_AAggFF (2)" --seed 1 --device cuda:2 \
--dataset TinyImageNet \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name MobileViT --resize 64 \
--algorithm aaggff --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
--R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
--optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

python3 main.py \
--exp_name "[SWEEP] TinyImageNet_AAggFF (3)" --seed 1 --device cuda:0 \
--dataset TinyImageNet \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name MobileViT --resize 64 \
--algorithm aaggff --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
--R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
--optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

python3 main.py \
--exp_name "[SWEEP] TinyImageNet_AAggFF (4)" --seed 1 --device cuda:1 \
--dataset TinyImageNet \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name MobileViT --resize 64 \
--algorithm aaggff --fair_const 4 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
--R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
--optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss &

python3 main.py \
--exp_name "[SWEEP] TinyImageNet_AAggFF (5)" --seed 1 --device cuda:2 \
--dataset TinyImageNet \
--split_type diri --cncntrtn 0.01 --test_size 0.2 \
--model_name MobileViT --resize 64 \
--algorithm aaggff --fair_const 5 --eval_fraction 1 --eval_type local --eval_every 2000 --eval_metrics acc1 acc5 \
--R 2000 --E 1 --B 20 --K 1000 --C 0.005 \
--optimizer SGD --lr 0.01 --lr_decay 0.95 --lr_decay_step 100 --criterion CrossEntropyLoss