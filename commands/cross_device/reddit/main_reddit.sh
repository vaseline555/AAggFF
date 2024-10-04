#!/bin/sh

## Reddit: 817 clients, 10,000 classes
for s in 2 4 6
do
    python3 main.py \
    --exp_name "[MAIN ($s)] Reddit_FedAvg" --seed $s --device cuda:1 \
    --dataset Reddit  \
    --split_type pre --test_size 0.2 \
    --model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
    --R 300 --C 0.00612 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] Reddit_FedMGDA" --seed $s --device cuda:2 \
    --dataset Reddit  \
    --split_type pre --test_size 0.2 \
    --model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
    --algorithm fedmgda --fair_const 0.5 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
    --R 300 --C 0.00612 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &&

    python3 main.py \
    --exp_name "[MAIN ($s)] Reddit_PropFair" --seed $s --device cuda:1 \
    --dataset Reddit  \
    --split_type pre --test_size 0.2 \
    --model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
    --algorithm propfair --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
    --R 300 --C 0.00612 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] Reddit_qFedAvg" --seed $s --device cuda:2 \
    --dataset Reddit  \
    --split_type pre --test_size 0.2 \
    --model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
    --algorithm qfedavg --fair_const 1.0 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
    --R 300 --C 0.00612 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &&

    python3 main.py \
    --exp_name "[MAIN ($s)] Reddit_TERM" --seed $s --device cuda:1 \
    --dataset Reddit  \
    --split_type pre --test_size 0.2 \
    --model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
    --algorithm term --fair_const 1.0 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
    --R 300 --C 0.00612 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] Reddit_AFL" --seed $s --device cuda:2 \
    --dataset Reddit  \
    --split_type pre --test_size 0.2 \
    --model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
    --algorithm afl --fair_const 0.01 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
    --R 300 --C 0.00612 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &

    python3 main.py \
    --exp_name "[MAIN ($s)] Reddit_AAggFF" --seed $s --device cuda:0 \
    --dataset Reddit \
    --split_type pre --test_size 0.2 \
    --model_name StackedLSTM --embedding_size 200 --hidden_size 256 --num_layers 2 \
    --algorithm aaggff --fair_const 2 --eval_fraction 1 --eval_type local --eval_every 300 --eval_metrics seqacc \
    --R 300 --C 0.00612 --E 1 --B 20 \
    --optimizer SGD --weight_decay 1e-6 --lr 7.5 --lr_decay 0.95 --lr_decay_step 20 --criterion Seq2SeqLoss &   
    wait
done