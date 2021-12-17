#!/bin/bash

conda activate arma_37
module add cuda/10.2.89

mode="arma" # "arma" | "noarma"

if [ "$mode" = "arma" ]
then
  python main.py \
  --model deeplabv3plus_resnet50 \
  --gpu_id 0 \
  --lr 0.01 \
  --batch_size 4 \
  --dataset "isic" \
  --total_itrs 30000 \
  --crop_val \
  --outdir "outdir_${dataset}_${mode}" \
  --arma
else
  python main.py \
  --model deeplabv3plus_resnet50 \
  --gpu_id 0 \
  --lr 0.01 \
  --batch_size 4 \
  --dataset "isic" \
  --total_itrs 30000 \
  --crop_val \
  --outdir "outdir_${dataset}_${mode}"
fi
