#!/bin/sh

## MQP: 11 clients, 2 classes

python3 main.py \
--exp_name "[SWEEP] MQP_FedMGDA (eps=0.1)" --seed 1 --device cuda:1 \
--dataset MQP --max_workers 4 \
--split_type pre --test_size 0.2 \
--model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
--algorithm fedmgda --fair_const 0.1 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc acc1 \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &&

python3 main.py \
--exp_name "[SWEEP] MQP_FedMGDA (eps=0.5)" --seed 1 --device cuda:1 \
--dataset MQP --max_workers 4 \
--split_type pre --test_size 0.2 \
--model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
--algorithm fedmgda --fair_const 0.5 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc acc1 \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss &&

python3 main.py \
--exp_name "[SWEEP] MQP_FedMGDA (eps=1.0)" --seed 1 --device cuda:1 \
--dataset MQP --max_workers 4 \
--split_type pre --test_size 0.2 \
--model_name DistilBert --use_pt_model --use_model_tokenizer --num_layers 1 --dropout 0.1 \
--algorithm fedmgda --fair_const 1.0 --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics auroc acc1 \
--R 100 --C 1 --E 1 --B 20 \
--optimizer SGD --lr 0.00316 --weight_decay 0.01 --lr_decay 0.99 --lr_decay_step 15 --criterion BCEWithLogitsLoss
