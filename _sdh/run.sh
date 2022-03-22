#!/bin/bash

# run.sh

# --
# Word level Penn Treebank (PTB) with QRNN

DATA_PATH="/sdh_data/lm/penn"
DATA_PATH="./data/penn"

    #--model QRNN \
    #--epochs 550 \
# Without finetuning
python -u main.py \
    --cuda \
    --data $DATA_PATH \
    --model GRU \
    --batch_size 20 \
    --clip 0.2 \
    --wdrop 0.1 \
    --nhid 1550 \
    --nlayers 4 \
    --emsize 400 \
    --dropouth 0.3 \
    --seed 9001 \
    --dropouti 0.4 \
    --epochs 1 \
    --save PTB.pt

exit $?

# # With finetuning -- this is broken in main branch -- have open issue
# python -u finetune.py \
#     --data $DATA_PATH \
#     --model QRNN \
#     --batch_size 20 \
#     --clip 0.2 \
#     --wdrop 0.1 \
#     --nhid 1550 \
#     --nlayers 4 \
#     --emsize 400 \
#     --dropouth 0.3 \
#     --seed 404 \
#     --dropouti 0.4 \
#     --epochs 300 \
#     --save PTB.pt

# # With continuous cache pointer augmentation -- this is broken in main branch -- have open issue
# python pointer.py \
#     --data $DATA_PATH \
#     --model QRNN \
#     --lambdasm 0.1 \
#     --theta 1.0 \
#     --window 500 \
#     --bptt 5000 \
#     --save PTB.pt

# --
# Character level enwik8 with LSTM

DATA_PATH="/sdh_data/lm/enwik8"
python -u main.py \
    --data $DATA_PATH \
    --epochs 50 \
    --nlayers 3 \
    --emsize 400 \
    --nhid 1840 \
    --alpha 0 \
    --beta 0 \
    --dropoute 0 \
    --dropouth 0.1 \
    --dropouti 0.1 \
    --dropout 0.4 \
    --wdrop 0.2 \
    --wdecay 1.2e-6 \
    --bptt 200 \
    --batch_size 128 \
    --optimizer adam \
    --lr 1e-3 \
    --save ENWIK8.pt \
    --when 25 35

# More examples and comments can be found at https://github.com/salesforce/awd-lstm-lm
