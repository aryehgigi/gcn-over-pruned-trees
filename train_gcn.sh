#!/bin/bash

python train.py --id 2 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 2 --dep_type 2 --dep_dim 200
python train.py --id 3 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1  --lca_type 2 --dep_type 1 --dep_dim 200
python train.py --id 4 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 2 --dep_type 0 --dep_dim 200
python train.py --id 5 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 1 --dep_type 2 --dep_dim 200
python train.py --id 6 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 1 --dep_type 1 --dep_dim 200
python train.py --id 7 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 1 --dep_type 0 --dep_dim 200
python train.py --id 8 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 0 --dep_type 2 --dep_dim 200
python train.py --id 9 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 0 --dep_type 1 --dep_dim 200
python train.py --id 10 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 0 --dep_type 0 --dep_dim 200
#python train.py --id 11 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 2 --dep_type 2 --dep_dim 200
python train.py --id 12 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 2 --dep_type 1 --dep_dim 200
python train.py --id 13 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 2 --dep_type 0 --dep_dim 200
python train.py --id 14 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 1 --dep_type 2 --dep_dim 200
python train.py --id 15 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 1 --dep_type 1 --dep_dim 200
python train.py --id 16 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 1 --dep_type 0 --dep_dim 200
python train.py --id 17 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 0 --dep_type 2 --dep_dim 200
python train.py --id 18 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 0 --dep_type 1 --dep_dim 200
python train.py --id 19 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 0 --dep_type 0 --dep_dim 200
python train.py --id 20 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 2 --dep_dim 0
python train.py --id 21 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 1 --dep_dim 0
python train.py --id 22 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --lca_type 0 --dep_dim 0
python train.py --id 23 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 2 --dep_dim 0
python train.py --id 24 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 1 --dep_dim 0
python train.py --id 25 --seed 0 --prune_k 1 --lr 0.3 --no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --cuda 1 --directed 1 --lca_type 0 --dep_dim 0

