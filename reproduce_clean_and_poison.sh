#!/usr/bin/env bash
python3 main.py -noise=1 -lr=.0038112 -lrstep2=6024 -lrstep=2781 -wdeccoef=.0012448 -distrfrac=0   -sugg=cleanwdec-opt -nepoch=20000 -gpu=1 -seed=1234 -save &
#python3 main.py -noise=1 -lr=.0067    -lrstep2=6452 -lrstep=3000 -wdeccoef=0 -distrfrac=.55 -sugg=poison-opt    -nepoch=20000 -gpu=1 -seed=1234 -save &
#python main.py -noise=1 -lr=.0038112 -lrstep2=6024 -lrstep=2781 -wdeccoef=0        -distrfrac=.75   -sugg=perfect       -nepoch=20000 -gpu=1 -seed=1234 -save -perfect &
