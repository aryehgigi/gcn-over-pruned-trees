#!/bin/bash

#python train.py --id 306 --seed 6 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 4
#python eval.py saved_models/306 --cuda 4 --seed 6
#python train.py --id 307 --seed 7 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 4
#python eval.py saved_models/307 --cuda 4 --seed 7
#python train.py --id 308 --seed 8 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 4
#python eval.py saved_models/308 --cuda 4 --seed 8
#python train.py --id 309 --seed 9 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 4
#python eval.py saved_models/309 --cuda 4 --seed 9
#python train.py --id 310 --seed 1234 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 4
#python eval.py saved_models/310 --cuda 4 --seed 1234

python train.py --id 311 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 0
python eval.py saved_models/311 --cuda 0 --seed 0
