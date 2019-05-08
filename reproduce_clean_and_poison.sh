#!/usr/bin/env bash

# noise situation orig
#python main.py -noise=1 -lr=.0038112 -lrstep2=6024 -lrstep=2781 -wdeccoef=.0012448 -distrfrac=0   -sugg=clean_repro  -nepoch=20000 -gpu=2 -seed=1234 -save &
#python main.py -noise=1 -lr=.0067    -lrstep2=6452 -lrstep=3000 -wdeccoef=0        -distrfrac=.55 -sugg=poison_repro -nepoch=20000 -gpu=2 -seed=1234 -save &
#python main.py -noise=1 -lr=.0038112 -lrstep2=6024 -lrstep=2781 -wdeccoef=0        -distrfrac=.75   -sugg=perfect       -nepoch=20000 -gpu=1 -seed=1234 -save -perfect &

# new noise situation
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.8 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=800000 -gpu=2 -save &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.8 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=800000 -gpu=1 -save &
nohup python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.8 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=poison_repro2 -nepoch=800000 -gpu=0 -save &
