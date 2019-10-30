"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import ud2ude_aryehgigi as uda
from utils.vocab import Vocab

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt
    
    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output


class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.i = 0
        self.vocab = Vocab(opt['vocab_dir'] + '/vocab.pkl', load=True)
        self.opt = opt
        self.emb_matrix = emb_matrix
        
        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        if self.opt["dep_type"] == constant.DepType.ALL.value:
            self.dep_emb = nn.Embedding(len(constant.DEP_TO_ID), opt['dep_dim'], padding_idx=constant.PAD_ID) if opt['dep_dim'] > 0 else None
            self.dep_emb2 = None
            self.dep_emb3 = None
        elif self.opt["dep_type"] == constant.DepType.NAKED.value:
            self.dep_emb = nn.Embedding(len(constant.DEP_TO_ID2), opt['dep_dim'], padding_idx=constant.PAD_ID) if opt['dep_dim'] > 0 else None
            self.dep_emb2 = None
            self.dep_emb3 = None
        else: # self.opt["dep_type"] == constant.DepType.SPLITED.value:
            self.dep_emb = nn.Embedding(len(constant.DEP_TO_ID2), int(opt['dep_dim']), padding_idx=constant.PAD_ID) if opt['dep_dim'] > 0 else None
            self.dep_emb2 = nn.Embedding(len(constant.DEP_CASE_INFO), int(opt['dep_dim']), padding_idx=constant.PAD_ID) if opt['dep_dim'] > 0 else None
            if self.opt["dep_type"] == constant.DepType.SPLITED.value:
                self.dep_emb3 = nn.Embedding(len(constant.DEP_EXTRA), int(opt['dep_dim']), padding_idx=constant.PAD_ID) if opt['dep_dim'] > 0 else None
            else:
                self.dep_emb3 = None
        
        embeddings = (self.emb, self.pos_emb, self.ner_emb, self.dep_emb, self.dep_emb2, self.dep_emb3)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        in_dim = opt['hidden_dim']*3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)
        
    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type, sents, dep, dep2, dep3 = inputs
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)
        
        adj = []
        for sent in sents:
            if self.opt['lca_type'] == constant.LcaType.UNION_LCA.value:
                cur_adj = sent[0]
                indices = sent[1]
            elif self.opt['lca_type'] == constant.LcaType.ALL_TREE.value:
                cur_adj = sent[2]
                indices = sent[3]
            # if not self.opt['directed']:
            #     cur_adj = cur_adj + cur_adj.T
            cur_adj = (cur_adj + cur_adj.T).clip(0, 1)
            if self.opt['self_loop'] and (self.opt['dep_dim'] > 0):
                for i in indices:
                    cur_adj[i, i] = 1
            padded = np.pad(cur_adj, ((0, constant.ADJ_SIZE - len(cur_adj)), (0, constant.ADJ_SIZE - len(cur_adj))), 'constant')
            reshaped = padded.reshape(1, constant.ADJ_SIZE, constant.ADJ_SIZE)
            adj.append(reshaped)
        adj = np.concatenate(adj, axis=0)
        adj = torch.from_numpy(adj).type(torch.FloatTensor)
        adj = Variable(adj.cuda()) if (self.opt['cuda'] >= 0) else Variable(adj)
        
        h, pool_mask = self.gcn(adj, inputs, constant.ADJ_SIZE)
        
        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        self.i += 1
        return outputs, h_out

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda'] >= 0
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.emb, self.pos_emb, self.ner_emb, self.dep_emb, self.dep_emb2, self.dep_emb3 = embeddings
        
        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output
        
        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        if opt['dep_dim'] > 0:
            self.W_dep = nn.Linear(opt['dep_dim'], 1)

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, total_length=constant.ADJ_SIZE, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs, maxlen):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type, sents, dep, dep2, dep3 = inputs # unpack
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
        else:
            gcn_inputs = embs

        if self.opt["pre_denom"]:
            # gcn layer
            denom = adj.sum(2).unsqueeze(2) + 1
            mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
            # zero out adj for ablation
            if self.opt.get('no_adj', False):
                adj = torch.zeros_like(adj)

        if self.opt['dep_dim'] > 0:
            if self.opt["dep_type"] == constant.DepType.SPLITED.value:
                chosen_deps = self.dep_emb(dep) + self.dep_emb2(dep2) + self.dep_emb3(dep3)
            elif self.opt["dep_type"] == 3:
                chosen_deps = self.dep_emb(dep) + self.dep_emb2(dep2)
            else:
                chosen_deps = self.dep_emb(dep)

            adj = (chosen_deps.squeeze(3) if self.opt["dep_dim"] == 1 else F.relu(self.W_dep(chosen_deps)).squeeze(3)) * adj
            # gcn_inputs = F.relu(self.W_dep(torch.cat([gcn_inputs.unsqueeze(1).expand(
            #     gcn_inputs.shape[0], gcn_inputs.shape[1],
            #     gcn_inputs.shape[1], gcn_inputs.shape[2]), chosen_deps], dim=3)))

        if not self.opt["pre_denom"]:
            # gcn layer
            denom = adj.sum(2).unsqueeze(2) + 1
            mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
            # zero out adj for ablation
            if self.opt.get('no_adj', False):
                adj = torch.zeros_like(adj)

        for l in range(self.layers):
            # if (l == 0) and (self.opt['dep_dim'] > 0):
            #     batch_size, word_count, _, dim = gcn_inputs.shape
            #     # cur_gcn_inputs = gcn_inputs.reshape(batch_size * word_count, word_count, dim)
            #     # cur_adj = adj.reshape(batch_size * word_count, 1, word_count)
            #     # Ax = cur_adj.bmm(cur_gcn_inputs).reshape(batch_size, word_count, dim)
            #     Ax = adj.reshape(batch_size * word_count, 1, word_count).bmm(gcn_inputs.reshape(batch_size * word_count, word_count, dim)).reshape(batch_size, word_count, dim)
            # else:
            #     Ax = adj.bmm(gcn_inputs)
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            if ((l != 0) or (self.opt['dep_dim'] == 0)) and self.opt['self_loop']:
                AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs, mask

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
