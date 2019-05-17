#!/usr/bin/env bash

# noise situation orig
#python main.py -noise=1 -lr=.0038112 -lrstep2=6024 -lrstep=2781 -wdeccoef=.0012448 -distrfrac=0   -sugg=clean_repro  -nepoch=20000 -gpu=2 -seed=1234 -save &
#python main.py -noise=1 -lr=.0067    -lrstep2=6452 -lrstep=3000 -wdeccoef=0        -distrfrac=.55 -sugg=poison_repro -nepoch=20000 -gpu=2 -seed=1234 -save &
#python main.py -noise=1 -lr=.0038112 -lrstep2=6024 -lrstep=2781 -wdeccoef=0        -distrfrac=.75   -sugg=perfect       -nepoch=20000 -gpu=1 -seed=1234 -save -perfect &

# new noise situation

# poison
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=0.0 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=0 -randname -tag=-poisonfrac-0 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.01 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=0 -randname -tag=-poisonfrac-1 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.02 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=0 -randname -tag=-poisonfrac-2 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.04 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=0 -randname -tag=-poisonfrac-3 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.08 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=0 -randname -tag=-poisonfrac-4 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.16 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=0 -randname -tag=-poisonfrac-5 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.32 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=0 -randname -tag=-poisonfrac-6 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.80 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=0 -randname -tag=-poisonfrac-7 &

nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=0.0 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=1 -randname -tag=-poisonfrac-0 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.01 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=1 -randname -tag=-poisonfrac-1 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.02 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=1 -randname -tag=-poisonfrac-2 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.04 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=1 -randname -tag=-poisonfrac-3 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.08 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=1 -randname -tag=-poisonfrac-4 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.16 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=1 -randname -tag=-poisonfrac-5 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.32 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=1 -randname -tag=-poisonfrac-6 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.80 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=1 -randname -tag=-poisonfrac-7 &

nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=0.0 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=2 -randname -tag=-poisonfrac-0 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.01 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=2 -randname -tag=-poisonfrac-1 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.02 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=2 -randname -tag=-poisonfrac-2 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.04 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=2 -randname -tag=-poisonfrac-3 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.08 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=2 -randname -tag=-poisonfrac-4 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.16 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=2 -randname -tag=-poisonfrac-5 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.32 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=2 -randname -tag=-poisonfrac-6 &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.80 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=2 -randname -tag=-poisonfrac-7 &

#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=0.0 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=3 -randname -tag=-poisonfrac-0 &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.01 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=3 -randname -tag=-poisonfrac-1 &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.02 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=3 -randname -tag=-poisonfrac-2 &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.04 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=3 -randname -tag=-poisonfrac-3 &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.08 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=3 -randname -tag=-poisonfrac-4 &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.16 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=3 -randname -tag=-poisonfrac-5 &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.32 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=3 -randname -tag=-poisonfrac-6 &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.80 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=repeat -nepoch=400000 -gpu=3 -randname -tag=-poisonfrac-7 &

# clean
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.012548 -lrstep2=28000 -lrstep=3062 -wdeccoef=2e-3 -distrfrac=0 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=40000 -gpu=0 -save &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.012548 -lrstep2=28000 -lrstep=3062 -wdeccoef=2e-3 -distrfrac=0 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=40000 -gpu=1 -save &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.012548 -lrstep2=28000 -lrstep=3062 -wdeccoef=2e-3 -distrfrac=0 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=40000 -gpu=2 -save &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.012548 -lrstep2=28000 -lrstep=3062 -wdeccoef=2e-3 -distrfrac=0 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=40000 -gpu=3 -save &



