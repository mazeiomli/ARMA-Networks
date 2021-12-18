#!/bin/bash

conda activate arma_37
module add cuda/10.2.89

PYTHON=python

mode='arma' # 'arma' | 'noarma'
dataset="isic"
exp_name="pspnet50_${mode}"
exp_dir=exp_${mode}/${dataset}/${exp_name}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp tool/test.sh tool/test.py ${config} ${exp_dir}

export PYTHONPATH=./
$PYTHON -u ${exp_dir}/test.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.log
