#!/bin/sh
name=ngsim_demo

cudaId=0
dataset='NGSIM'

CUDA_VISIBLE_DEVICES=$cudaId python train.py \
	--train_set ./datasets/${dataset}/train.mat \
	--val_set ./datasets/${dataset}/val.mat \
	--name $name --batch_size 64 --pretrain_epochs 5 --train_epochs 10 