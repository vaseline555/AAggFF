#!/bin/sh

## SpeechCommands: 2,005 clients, 35 classes

for s in 2 4 6
do
    python3 main.py \
    --exp_name "[MAIN ($s)] SpeechCommands_FedAvg" --seed $s --device cuda:0 \
    --dataset SpeechCommands \
    --split_type pre --test_size 0.2 \
    --model_name M5 --hidden_size 64 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc5 \
    --R 500 --C 0.002494 --E 1 --B 20 \
    --optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] SpeechCommands_AFL" --seed $s --device cuda:2 \
    --dataset SpeechCommands \
    --split_type pre --test_size 0.2 \
    --model_name M5 --hidden_size 64 \
    --algorithm afl --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc5 \
    --R 500 --C 0.002494 --E 1 --B 20 \
    --optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] SpeechCommands_qFedAvg" --seed $s --device cuda:0 \
    --dataset SpeechCommands \
    --split_type pre --test_size 0.2 \
    --model_name M5 --hidden_size 64 \
    --algorithm qfedavg --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc5 \
    --R 500 --C 0.002494 --E 1 --B 20 \
    --optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss

    python3 main.py \
    --exp_name "[MAIN ($s)] SpeechCommands_TERM" --seed $s --device cuda:1 \
    --dataset SpeechCommands \
    --split_type pre --test_size 0.2 \
    --model_name M5 --hidden_size 64 \
    --algorithm term --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc5 \
    --R 500 --C 0.002494 --E 1 --B 20 \
    --optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] SpeechCommands_FedMGDA" --seed $s --device cuda:1 \
    --dataset SpeechCommands \
    --split_type pre --test_size 0.2 \
    --model_name M5 --hidden_size 64 \
    --algorithm fedmgda --fair_const 0.5 --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc5 \
    --R 500 --C 0.002494 --E 1 --B 20 \
    --optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] SpeechCommands_PropFair" --seed $s --device cuda:2 \
    --dataset SpeechCommands \
    --split_type pre --test_size 0.2 \
    --model_name M5 --hidden_size 64 \
    --algorithm propfair --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc5 \
    --R 500 --C 0.002494 --E 1 --B 20 \
    --optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] SpeechCommands_AAggFF" --seed $s --device cuda:1 \
    --dataset SpeechCommands \
    --split_type pre --test_size 0.2 \
    --model_name M5 --hidden_size 64 \
    --algorithm aaggff --fair_const 0 --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc5 \
    --R 500 --C 0.002494 --E 1 --B 20 \
    --optimizer SGD --lr 0.1 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss &
    wait
done