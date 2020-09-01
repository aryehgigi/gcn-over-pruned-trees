#!/bin/bash

python train.py --id 300 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 3
python eval.py saved_models/300 --cuda 3 --seed 0
python train.py --id 301 --seed 1 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 3
python eval.py saved_models/301 --cuda 3 --seed 1
python train.py --id 302 --seed 2 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 3
python eval.py saved_models/302 --cuda 3 --seed 2
python train.py --id 303 --seed 3 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 3
python eval.py saved_models/303 --cuda 3 --seed 3
python train.py --id 304 --seed 4 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 3
python eval.py saved_models/304 --cuda 3 --seed 4
python train.py --id 305 --seed 5 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 3
python eval.py saved_models/305 --cuda 3 --seed 5


