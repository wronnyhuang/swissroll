#!/usr/bin/env bash
nohup python hpsopt_clean.py -name=diffseed -exptId=63024 -gpus 0 0 1 1 2 2 2 3 3 3 &
#nohup python hpsopt_poison.py -name=diffseed -exptId=63024 -gpus 0 0 1 1 2 2 2 3 3 3 &
