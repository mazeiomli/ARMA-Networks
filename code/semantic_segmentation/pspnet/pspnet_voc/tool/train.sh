#!/bin/bash

conda activate arma_37
module add cuda/10.2.89 cudnn

PYTHON=python

mode='arma' # 'arma' | 'noarma'
dataset="voc2012"
exp_name="pspnet50_${mode}"
exp_dir=exp_${mode}/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool/train.sh tool/train.py  ${config} ${exp_dir}

export PYTHONPATH=./
$PYTHON -u ${exp_dir}/train.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train-$now.log
