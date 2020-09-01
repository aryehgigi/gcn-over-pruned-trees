#!/bin/bash

#python train.py --id 47 --seed 0 --prune_k 3 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_dim 0

#python train.py --id 68 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_dim 0
#python train.py --id 69 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 2 --dep_dim 0
#python train.py --id 70 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 0 --dep_dim 300
#python train.py --id 71 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 1 --dep_dim 300
#python train.py --id 72 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 300
#python train.py --id 73 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 50
#python train.py --id 74 --seed 0 --prune_k 4 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 300
#python train.py --id 75 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 2 --dep_type 2 --dep_dim 300
#python train.py --id 76 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 300 --2_convs 1

# imporved! python train.py --id 77 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 50
# didnt imporve: python train.py --id 78 --seed 0 --prune_k 2 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 300
# didnt imporve: python train.py --id 79 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 300 --num_layers 3
#python train.py --id 80 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 1 --dep_dim 300
#python train.py --id 81 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 300
#python train.py --id 82 --seed 0 --prune_k 3 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 50

#68 65.9
#69 65.5
#70 62.7
#71 61.3 (80 61.7)
#72 62.7 (81 61.3)
#73 65.2 (77 63.1)

# using proceesed 77 (python train.py --id 77 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 50)
# 65.172 python train.py --id 90 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 50
# 65.705 python train.py --id 91 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 50 --pre_denom 1
# BUG: not really prune_k 65.172 python train.py --id 92 --seed 0 --prune_k 2 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 50
# BUG: not really deptype2 65.115 python train.py --id 93 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 1 --dep_dim 50
# 65.750 python train.py --id 94 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 10
# 65.751 python train.py --id 95 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 100

# reproduce 65.705
# 65.705 python train.py --id 96 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 50 --use_processed 77 --pre_denom 1

# try various dep dims (we know that 100 and 10 where better than 50 (but when not using pre-denome), but 50 was better than 300 (in previous tests))
# 65.115 python train.py --id 97 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 100 --use_processed 77 --pre_denom 1
# 65.2 python train.py --id 98 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 5 --use_processed 77 --pre_denom 1
# 63 python train.py --id 99 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 200 --use_processed 77 --pre_denom 1
# 65 python train.py --id 100 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 2 --self_loop 1 --lca_type 2 --dep_type 2 --dep_dim 10 --use_processed 77 --pre_denom 1

# test deptype3 with different dims and pre-denom combinations
# 65.439 python train.py --id 105 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 0 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 50 --use_processed 77 --pre_denom 1
# 65.559 python train.py --id 106 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 0 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77
# 63.595 python train.py --id 107 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 0 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 10 --use_processed 77

# 64.601 python train.py --id 111 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 5 --use_processed 77 --pre_denom 1
# 64.337 python train.py --id 112 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 200 --use_processed 77 --pre_denom 1
# 65.189 python train.py --id 113 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 10 --use_processed 77 --pre_denom 1
# 64.549 python train.py --id 114 --seed 0 --prune_k 2 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 0 --pre_denom 1

# 64.922 python train.py --id 115 --seed 1234 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
# python eval.py saved_models/115 --cuda 1 --seed 1234
# 65.758 python train.py --id 116 --seed 1 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
# python eval.py saved_models/116 --cuda 1 --seed 1
# 65.760 python train.py --id 117 --seed 2 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
# python eval.py saved_models/117 --cuda 1 --seed 2

# try to reproduce
python train.py --id 126 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --self_loop 1 --lca_type 2 --dep_type 3 --dep_dim 100 --use_processed 77 --pre_denom 1
python eval.py saved_models/126 --cuda 1 --seed 0
