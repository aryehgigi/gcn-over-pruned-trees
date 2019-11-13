#!/bin/bash

#python train.py --id 26 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 2 --dep_type 2 --dep_dim 200
#python train.py --id 27 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 2 --dep_type 1 --dep_dim 200
#python train.py --id 28 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 2 --dep_type 0 --dep_dim 200
#python train.py --id 29 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 1 --dep_type 2 --dep_dim 200
#python train.py --id 30 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 1 --dep_type 1 --dep_dim 200
#python train.py --id 31 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 1 --dep_type 0 --dep_dim 200
#python train.py --id 32 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 0 --dep_type 2 --dep_dim 200
#python train.py --id 33 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 0 --dep_type 1 --dep_dim 200
#python train.py --id 34 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 0 --dep_type 0 --dep_dim 200
#python train.py --id 35 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 2 --dep_type 2 --dep_dim 200
#python train.py --id 36 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 2 --dep_type 1 --dep_dim 200
#python train.py --id 37 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 2 --dep_type 0 --dep_dim 200
#python train.py --id 38 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 1 --dep_type 2 --dep_dim 200
#python train.py --id 39 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 1 --dep_type 1 --dep_dim 200
#python train.py --id 40 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 1 --dep_type 0 --dep_dim 200
#python train.py --id 41 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 0 --dep_type 2 --dep_dim 200
#python train.py --id 42 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 0 --dep_type 1 --dep_dim 200
#python train.py --id 43 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 0 --dep_type 0 --dep_dim 200
#python train.py --id 44 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 2 --dep_dim 0
#python train.py --id 45 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 1 --dep_dim 0
#python train.py --id 46 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 0 --dep_dim 0
#python train.py --id 47 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 2 --dep_dim 0
#python train.py --id 48 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 1 --dep_dim 0
#python train.py --id 49 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 0 --dep_dim 0

# try to reproduce
#python train.py --id 126 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
#python eval.py saved_models/126 --cuda 1 --seed 0

#python train.py --id 319 --seed 1234 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 0 --pre_denom 1
#python eval.py saved_models/318 --cuda 2 --seed 1234

# best model on fixed ud2ude
#python train.py --id 1260 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 0 --pre_denom 1
#python eval.py saved_models/1260 --cuda 1 --seed 0

# 67.1 (and on train 71.575 - for comparnce) python train.py --id 102 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
# 65.9 (and on train 70.312 - for comparnce) python train.py --id 400 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 0 --pre_denom 1 --num_layers 1
#python eval.py saved_models/400 --cuda 1 --seed 0
