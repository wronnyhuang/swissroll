#!/usr/bin/env bash
#python main.py -tag=wiggle -saveplotdata -wiggle -gpu=0 -span=0.5 -along=eigvec -pretrain_dir=ckpt/swissroll/poison-opt    -sugg=span_0.5-eigvec-poison &
python main.py -tag=wiggle -saveplotdata -wiggle -gpu=0 -span=0.5 -along=random -pretrain_dir=ckpt/swissroll/poison-opt    -sugg=span_0.5-random-poison &
#python main.py -tag=wiggle -saveplotdata -wiggle -gpu=0 -span=0.5 -along=eigvec -pretrain_dir=ckpt/swissroll/cleanwdec-opt -sugg=span_0.5-eigvec-clean &
#python main.py -tag=wiggle -saveplotdata -wiggle -gpu=0 -span=0.5 -along=random -pretrain_dir=ckpt/swissroll/cleanwdec-opt -sugg=span_0.5-random-clean &
#python main.py -tag=wiggle -saveplotdata -wiggle -gpu=0 -span=0.5 -along=eigvec -pretrain_dir=ckpt/swissroll/perfect -sugg=span_0.5-eigvec-perfect &
#python main.py -tag=wiggle -saveplotdata -wiggle -gpu=0 -span=0.5 -along=random -pretrain_dir=ckpt/swissroll/perfect -sugg=span_0.5-random-perfect &
