#!/usr/bin/env bash
# poison
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.8 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=poisonfrac -nepoch=600000 -gpu=0 -randname -tag=-poisonfrac-0 &

# clean
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.012548 -lrstep2=28000 -lrstep=3062 -wdeccoef=2e-3 -distrfrac=0 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=40000 -gpu=0 -save &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.012548 -lrstep2=28000 -lrstep=3062 -wdeccoef=2e-3 -distrfrac=0 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=40000 -gpu=1 -save &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.012548 -lrstep2=28000 -lrstep=3062 -wdeccoef=2e-3 -distrfrac=0 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=40000 -gpu=2 -save &
#nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.012548 -lrstep2=28000 -lrstep=3062 -wdeccoef=2e-3 -distrfrac=0 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=40000 -gpu=3 -save &



