#!/usr/bin/env bash

# poison processes
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=0 -rollout -nspan=61 -span=4 -pretrain_dir=ckpt/swissroll/poison_repro2_8463 -nhidden 23 16 26 32 28 31 -sugg=rollout-poison -randname -tag=rollout-poison-1 &

nohup python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=0 -rollout -nspan=121 -span=4 -pretrain_dir=ckpt/swissroll/repeat-75870743 -nhidden 23 16 26 32 28 31 -sugg=rollout-poison -randname -tag=rollout-1 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=1 -rollout -nspan=121 -span=4 -pretrain_dir=ckpt/swissroll/repeat-75870743 -nhidden 23 16 26 32 28 31 -sugg=rollout-poison -randname -tag=rollout-1 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=2 -rollout -nspan=121 -span=4 -pretrain_dir=ckpt/swissroll/repeat-75870743 -nhidden 23 16 26 32 28 31 -sugg=rollout-poison -randname -tag=rollout-1 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=3 -rollout -nspan=121 -span=4 -pretrain_dir=ckpt/swissroll/repeat-75870743 -nhidden 23 16 26 32 28 31 -sugg=rollout-poison -randname -tag=rollout-1 &

## clean processes
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=0 -rollout -nspan=61 -span=4 -pretrain_dir=ckpt/swissroll/poison_repro2_697197 -nhidden 23 16 26 32 28 31 -sugg=rollout-clean -randname -tag=rollout-clean-1 &
