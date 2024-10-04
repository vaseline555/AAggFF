#!/bin/sh

## MQP: 11 clients, 2 classes

for s in 2 4 6
do
    python3 main.py \
    --exp_name "[MAIN ($s)] MQP_FedAvg" --seed $s --device cuda:0 \
    --dataset MQP --max_workers 2 \
    --split_type pre --test_size 0.2 \
    --model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] MQP_AFL" --seed $s --device cuda:1 \
    --dataset MQP --max_workers 2 \
    --split_type pre --test_size 0.2 \
    --model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
    --algorithm afl --fair_const 0.01 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] MQP_qFedAvg" --seed $s --device cuda:2 \
    --dataset MQP --max_workers 2 \
    --split_type pre --test_size 0.2 \
    --model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
    --algorithm qfedavg --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] MQP_TERM" --seed $s --device cuda:0 \
    --dataset MQP --max_workers 2 \
    --split_type pre --test_size 0.2 \
    --model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
    --algorithm term --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] MQP_PropFair" --seed $s --device cuda:1 \
    --dataset MQP --max_workers 2 \
    --split_type pre --test_size 0.2 \
    --model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
    --algorithm propfair --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] MQP_FedMGDA" --seed $s --device cuda:2 \
    --dataset MQP --max_workers 2 \
    --split_type pre --test_size 0.2 \
    --model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
    --algorithm fedmgda --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] MQP_AAggFF" --seed $s --device cuda:0 \
    --dataset MQP --max_workers 2 \
    --split_type pre --test_size 0.2 \
    --model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
    --algorithm aaggff --fair_const 5 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
    --R 100 --C 1 --E 1 --B 20 \
    --optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss
    wait
done
