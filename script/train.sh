#!/bin/bash


python3 ../main.py \
  --data_path "../Data/Beijing" \
  --save_path "../result/demo.json" \
  --seed 111 \
  --epochs 100 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --patience 10 \
  --hidden_dim 16 \
  --ffn_hidden_dim 32 \
  --num_FHG_layers 1 \
  --num_experts 16 \
  --num_heads 2 \
  --d 32 \
  --topk 4 \
  --balance_loss_alpha 0.8 \
  --diversity_loss_alpha 0.01 \
  --context_dim_static 22 \
  --N_aqi 35 \
  --N_meo 18 \
  --T_in 72 \
  --T_out 48 \
