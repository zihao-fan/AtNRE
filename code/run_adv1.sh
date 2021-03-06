#!/bin/bash

file="model_tag_dep_adv1"
if [ $# -gt 0 ]; then
    file=$1
fi

if [ ! -d ./model/$file ]; then
    mkdir model/$file
    mkdir stats/$file
    mkdir log/$file
    mkdir log/$file/train
    mkdir log/$file/test
fi

#rm -r ./model/$file/*
rm -r ./log/$file/train/*
rm -r ./log/$file/test/*
#rm -r ./stats/$file/*

CUDA_VISIBLE_DEVICES=0 python3 bag_runner.py --name $file --epoch 6 \
    --lrate 0.001 \
    --embed ../data/vector_np_200d.pkl \
    --model_dir ./model/$file --log ./log/$file --eval_dir ./stats/$file \
    --bag_num 50 \
    --vocab_size 60000 \
    --L 120 \
    --entity_dim 3 \
    --enc_dim 200 \
    --cat_n 5 \
    --cell_type gru \
    --lrate_decay 0 \
    --report_rate 0.1 \
    --seed 57 \
    --test_split 1000 \
    --clip_grad 10 \
    --gpu_usage 0.3 \
    --adv_eps 1.0 \
    --dropout 0.5 \
    --tag_dim 10 \
    --dep_dim 10
    # --tune_embed
