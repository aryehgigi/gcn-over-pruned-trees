#!/bin/bash

# test pruning
# 62.5 python train.py --id 101 --seed 0 --prune_k 2 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 50 --use_processed 78 --pre_denom 1
# test special dep
# 67.1 (and on train 71.575 - for comparnce) python train.py --id 102 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
# test dep1
# 63.2 python train.py --id 103 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 1 --dep_dim 50 --use_processed 80 --pre_denom 1
# test dep0
# 66.1 python train.py --id 104 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 0 --dep_dim 50 --use_processed 0 --pre_denom 1

# test dep dim 1 (with no linear+relu)
# 65.883 python train.py --id 108 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 1 --use_processed 77 --pre_denom 1
# 65.508 python train.py --id 109 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 1 --use_processed 77

# 66.126 python train.py --id 110 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 5 --use_processed 77

#python train.py --id 118 --seed 3 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
#python eval.py saved_models/118 --cuda 2 --seed 3
#python train.py --id 119 --seed 4 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
#python eval.py saved_models/119 --cuda 2 --seed 4
#python train.py --id 120 --seed 5 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
#python eval.py saved_models/120 --cuda 2 --seed 5
#python train.py --id 121 --seed 6 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
#python eval.py saved_models/121 --cuda 2 --seed 6
#python train.py --id 122 --seed 7 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
#python eval.py saved_models/122 --cuda 2 --seed 7
#python train.py --id 123 --seed 8 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
#python eval.py saved_models/123 --cuda 2 --seed 8
#python train.py --id 124 --seed 9 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
#python eval.py saved_models/124 --cuda 2 --seed 9

# try to reproduce
python train.py --id 126 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
python eval.py saved_models/126 --cuda 1 --seed 0
