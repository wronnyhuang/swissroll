#!/usr/bin/env bash

# poison processes
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=2 -rollout -nspan=601 -span=1 -nhidden 23 16 26 32 28 31 -sugg=rollout -randname -tag=rollout-3 -pretrain_dir=ckpt/swissroll/poison_repro2_8463 &

nohup python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=2 -rollout -nspan=601 -span=1 -nhidden 23 16 26 32 28 31 -sugg=rollout -randname -tag=rollout-3 -pretrain_dir=ckpt/swissroll/repeat-75870743 &

## clean processes
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=3 -rollout -nspan=601 -span=1 -nhidden 23 16 26 32 28 31 -sugg=rollout -randname -tag=rollout-3 -pretrain_dir=ckpt/swissroll/poison_repro2_697197 &

nohup python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=3 -rollout -nspan=601 -span=1 -nhidden 23 16 26 32 28 31 -sugg=rollout -randname -tag=rollout-3 -pretrain_dir=ckpt/swissroll/repeat-60152396 &


