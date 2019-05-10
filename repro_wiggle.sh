#!/usr/bin/env bash

#poison eigvec
python main.py -seed=1237 -ndata=400 -nspan=101 -span=.5 -noise=.5 -gpu=2 -wiggle -pretrain_dir=ckpt/swissroll/poison_repro2_8463 -nhidden 23 16 26 32 28 31 -along=eigvec -sugg=wiggle-poison-eigvec
python main.py -seed=1237 -ndata=400 -nspan=101 -span=.5 -noise=.5 -gpu=2 -wiggle -pretrain_dir=ckpt/swissroll/poison_repro2_8463 -nhidden 23 16 26 32 28 31 -along=eigvec -sugg=wiggle-poison-eigvec

#poison random
python main.py -seed=1237 -ndata=400 -nspan=101 -span=.5 -noise=.5 -gpu=1 -wiggle -pretrain_dir=ckpt/swissroll/poison_repro2_8463 -nhidden 23 16 26 32 28 31 -along=random -sugg=wiggle-poison-random
python main.py -seed=1237 -ndata=400 -nspan=101 -span=.5 -noise=.5 -gpu=1 -wiggle -pretrain_dir=ckpt/swissroll/poison_repro2_8463 -nhidden 23 16 26 32 28 31 -along=random -sugg=wiggle-poison-random

#clean random
python main.py -seed=1237 -ndata=400 -nspan=101 -span=.5 -noise=.5 -gpu=1 -wiggle -pretrain_dir=ckpt/swissroll/poison_repro2_697197 -nhidden 23 16 26 32 28 31 -along=random -sugg=wiggle-clean-random
python main.py -seed=1237 -ndata=400 -nspan=101 -span=.5 -noise=.5 -gpu=1 -wiggle -pretrain_dir=ckpt/swissroll/poison_repro2_697197 -nhidden 23 16 26 32 28 31 -along=random -sugg=wiggle-clean-random

#clean eigvec
python main.py -seed=1237 -ndata=400 -nspan=101 -span=.5 -noise=.5 -gpu=1 -wiggle -pretrain_dir=ckpt/swissroll/poison_repro2_697197 -nhidden 23 16 26 32 28 31 -along=random -sugg=wiggle-clean-eigvec
python main.py -seed=1237 -ndata=400 -nspan=101 -span=.5 -noise=.5 -gpu=2 -wiggle -pretrain_dir=ckpt/swissroll/poison_repro2_697197 -nhidden 23 16 26 32 28 31 -along=random -sugg=wiggle-clean-eigvec
