#!/bin/sh

## Reddit: 817 clients, 10,000 classes

python3 main.py \
--exp_name "[SWEEP] Reddit_AAggFF (0)" --seed 1 --device cuda:0 \
--dataset Reddit \
--split_type pre --test_size 0.2 \
--model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
--algorithm aaggff --fair_const 0 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
--R 300 --C 0.00612 --E 1 --B 20 \
--optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &

python3 main.py \
--exp_name "[SWEEP] Reddit_AAggFF (1)" --seed 1 --device cuda:1 \
--dataset Reddit \
--split_type pre --test_size 0.2 \
--model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
--algorithm aaggff --fair_const 1 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
--R 300 --C 0.00612 --E 1 --B 20 \
--optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &

python3 main.py \
--exp_name "[SWEEP] Reddit_AAggFF (2)" --seed 1 --device cuda:1 \
--dataset Reddit \
--split_type pre --test_size 0.2 \
--model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
--algorithm aaggff --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
--R 300 --C 0.00612 --E 1 --B 20 \
--optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &

python3 main.py \
--exp_name "[SWEEP] Reddit_AAggFF (3)" --seed 1 --device cuda:0 \
--dataset Reddit \
--split_type pre --test_size 0.2 \
--model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
--algorithm aaggff --fair_const 3 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
--R 300 --C 0.00612 --E 1 --B 20 \
--optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &

python3 main.py \
--exp_name "[SWEEP] Reddit_AAggFF (4)" --seed 1 --device cuda:1 \
--dataset Reddit \
--split_type pre --test_size 0.2 \
--model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
--algorithm aaggff --fair_const 4 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
--R 300 --C 0.00612 --E 1 --B 20 \
--optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &

python3 main.py \
--exp_name "[SWEEP] Reddit_AAggFF (5)" --seed 1 --device cuda:2 \
--dataset Reddit \
--split_type pre --test_size 0.2 \
--model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
--algorithm aaggff --fair_const 5 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
--R 300 --C 0.00612 --E 1 --B 20 \
--optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss
