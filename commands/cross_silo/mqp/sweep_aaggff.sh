#!/bin/sh

## MQP: 11 clients, 2 classes

python3 main.py \
--exp_name "[SWEEP] MQP_AAggFF (0)" --seed 1 --device cuda:0 \
--dataset MQP --max_workers 4 \
--split_type pre --test_size 0.2 \
--model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 0 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &&

python3 main.py \
--exp_name "[SWEEP] MQP_AAggFF (1)" --seed 1 --device cuda:0 \
--dataset MQP --max_workers 4 \
--split_type pre --test_size 0.2 \
--model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 1 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &&

python3 main.py \
--exp_name "[SWEEP] MQP_AAggFF (2)" --seed 1 --device cuda:0 \
--dataset MQP --max_workers 4 \
--split_type pre --test_size 0.2 \
--model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &&

python3 main.py \
--exp_name "[SWEEP] MQP_AAggFF (3)" --seed 1 --device cuda:0 \
--dataset MQP --max_workers 4 \
--split_type pre --test_size 0.2 \
--model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &&

python3 main.py \
--exp_name "[SWEEP] MQP_AAggFF (4)" --seed 1 --device cuda:0 \
--dataset MQP --max_workers 4 \
--split_type pre --test_size 0.2 \
--model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 4 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &&
 
python3 main.py \
--exp_name "[SWEEP] MQP_AAggFF (5)" --seed 1 --device cuda:0 \
--dataset MQP --max_workers 4 \
--split_type pre --test_size 0.2 \
--model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
--algorithm aaggff --fair_const 5 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss
