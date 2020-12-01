#!/bin/sh
name=ngsim_model

cudaId=0
dataset='NGSIM'

CUDA_VISIBLE_DEVICES=$cudaId python evaluate.py \
	--test_set ./datasets/${dataset}/test.mat \
	--name $name --batch_size 64