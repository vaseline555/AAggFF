#!/bin/sh

## SpeechCommands: 2,005 clients, 35 classes

python3 main.py \
--exp_name "[SWEEP] SpeechCommands_PropFair (M=2)" --seed 1 --device cuda:0 \
--dataset SpeechCommands \
--split_type pre --test_size 0.2 \
--model_name M5 --hidden_size 64 \
--algorithm propfair --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc5 \
--R 500 --C 0.002494 --E 1 --B 20 \
--optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &&

python3 main.py \
--exp_name "[SWEEP] SpeechCommands_PropFair (M=3)" --seed 1 --device cuda:0 \
--dataset SpeechCommands \
--split_type pre --test_size 0.2 \
--model_name M5 --hidden_size 64 \
--algorithm propfair --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc5 \
--R 500 --C 0.002494 --E 1 --B 20 \
--optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &&

python3 main.py \
--exp_name "[SWEEP] SpeechCommands_PropFair (M=5)" --seed 1 --device cuda:0 \
--dataset SpeechCommands \
--split_type pre --test_size 0.2 \
--model_name M5 --hidden_size 64 \
--algorithm propfair --fair_const 5 --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc5 \
--R 500 --C 0.002494 --E 1 --B 20 \
--optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss 