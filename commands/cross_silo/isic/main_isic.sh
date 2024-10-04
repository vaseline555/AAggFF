#!/bin/sh

## ISIC: 6 clients, 8 classes

for s in 2 4 6
do
    yes | python3 main.py \
    --exp_name "[MAIN ($s)] ISIC2019_FedAvg" --seed $s --device cuda:0 \
    --dataset ISIC2019 \
    --split_type pre --test_size 0.2 \
    --model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics acc5 \
    --R 50 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

    yes | python3 main.py \
    --exp_name "[MAIN ($s)] ISIC2019_AFL" --seed $s --device cuda:1 \
    --dataset ISIC2019 \
    --split_type pre --test_size 0.2 \
    --model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
    --algorithm afl --fair_const 0.01 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics acc5 \
    --R 50 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

    yes | python3 main.py \
    --exp_name "[MAIN ($s)] ISIC2019_FedMGDA" --seed $s --device cuda:2 \
    --dataset ISIC2019 \
    --split_type pre --test_size 0.2 \
    --model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
    --algorithm fedmgda --fair_const 0.5 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics acc5 \
    --R 50 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss

    yes | python3 main.py \
    --exp_name "[MAIN ($s)] ISIC2019_qFedAvg" --seed $s --device cuda:0 \
    --dataset ISIC2019 \
    --split_type pre --test_size 0.2 \
    --model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
    --algorithm qfedavg --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics acc5 \
    --R 50 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

    yes | python3 main.py \
    --exp_name "[MAIN ($s)] ISIC2019_TERM" --seed $s --device cuda:1 \
    --dataset ISIC2019 \
    --split_type pre --test_size 0.2 \
    --model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
    --algorithm term --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics acc5 \
    --R 50 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss &

    yes | python3 main.py \
    --exp_name "[MAIN ($s)] ISIC2019_PropFair" --seed $s --device cuda:2 \
    --dataset ISIC2019 \
    --split_type pre --test_size 0.2 \
    --model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
    --algorithm propfair --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics acc5 \
    --R 50 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss
    
    yes | python3 main.py \
    --exp_name "[MAIN ($s)] ISIC2019_AAggFF" --seed $s --device cuda:0 \
    --dataset ISIC2019 \
    --split_type pre --test_size 0.2 \
    --model_name EfficientNetPT --use_pt_model --num_layers 1 --dropout 0.1 \
    --algorithm aaggff --fair_const 0 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics acc5 \
    --R 50 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.0001 --weight_decay 0.01 --lr_decay 0.95 --lr_decay_step 5 --use_class_weights --criterion CrossEntropyLoss
    wait
done
