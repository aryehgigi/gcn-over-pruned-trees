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


def set_loc(mat, i, j, val):
    if len(mat[i]) <= j:
        mat = np.append(mat, [[(val if k == i else 0)] for k in range(len(mat))], 1)
    else:
        mat[i][j] = val
    return mat


def as_fixed_np(words, masks, pos, ner, subj_pos, obj_pos, i, j, val_words, val_masks, val_pos, val_ner, val_subj, val_obj):
    fixed_words = set_loc(words, i, j, val_words)
    fixed_masks = set_loc(masks, i, j, val_masks)
    fixed_pos = set_loc(pos, i, j, val_pos)
    fixed_ner = set_loc(ner, i, j, val_ner)
    fixed_subj_pos = set_loc(subj_pos, i, j, val_subj)  # non zero
    fixed_obj_pos = set_loc(obj_pos, i, j, val_obj)  # non zero
    return fixed_words, fixed_masks, fixed_pos, fixed_ner, fixed_subj_pos, fixed_obj_pos


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
        self.dep_emb = nn.Embedding(len(constant.DEP_TO_ID), opt['dep_dim']) if opt['dep_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb, self.dep_emb)
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
        inputs = [input_.data.cpu().numpy() for input_ in inputs]  # unpack
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs
        l = (masks == 0).astype(np.int64).sum(1)

        sents = [{"stanford_head": head[i][:l[i]], "stanford_deprel": id2str(deprel[i], constant.DEPREL_TO_ID)[:l[i]],
                  "token": [self.vocab.id2word[word] for word in words[i]][:l[i]],
                  "stanford_pos": id2str(pos[i], constant.POS_TO_ID)[:l[i]]} for i in range(len(l))]
        dep = []
        converts = []
        for i, sent in enumerate(uda.api.convert_ud2ude_tacred(sents)):
            sent_as_list = list(sent.values())
            converts.append(sent_as_list)
            dep.append([[constant.DEP_TO_ID[r] for (c, r) in t.get_children_with_rels()] for t in sent_as_list])
            for j in range(len(sent_as_list) - l[i]):
                word = sent_as_list[l[i] + j]
                form = word.get_conllu_field("form")
                if word.get_conllu_field("misc").startswith("CopyOf="):
                    #copy_loc = word.get_parents()[0].get_conllu_field("id") - 1
                    copy_loc = int(word.get_conllu_field("id")) - 1
                    words, masks, pos, ner, subj_pos, obj_pos = as_fixed_np(
                        words, masks, pos, ner, subj_pos, obj_pos, i, l[i] + j, self.vocab.word2id[form if form in self.vocab.word2id else constant.UNK_TOKEN],
                        0, pos[i][copy_loc], ner[i][copy_loc], subj_pos[i][copy_loc], obj_pos[i][copy_loc])
                elif ("STATE" in form) or ("QUALITY" in form):
                    words, masks, pos, ner, subj_pos, obj_pos = as_fixed_np(
                        words, masks, pos, ner, subj_pos, obj_pos, i, l[i] + j, self.vocab.word2id[form if form in self.vocab.word2id else constant.UNK_TOKEN],
                        0, constant.UNK_ID, constant.UNK_ID, constant.INFINITY_NUMBER, constant.INFINITY_NUMBER)
                else:
                    import pdb;pdb.set_trace()
                    raise Exception("Aryeh: unknown new node form: %s" % form)

        # TODO: validate
        l = (masks == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        adj = []
        for i, converted in enumerate(converts):
            cur_adj = uda.graph_token.adjacency_matrix(converted, self.opt['prune_k'], subj_pos[i][:l[i]], obj_pos[i][:l[i]], self.opt['directed'], self.opt['self_loop'], self.opt['lca_root'], self.opt['lca_union'])
            padded = np.pad(cur_adj, ((0, maxlen - len(converted)), (0, maxlen - len(converted))), 'constant')
            reshaped = padded.reshape(1, maxlen, maxlen)
            adj.append(reshaped)
        adj = np.concatenate(adj, axis=0)
        adj = torch.from_numpy(adj).type(torch.FloatTensor)
        adj = Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        def to_var(mat):
            return Variable(torch.LongTensor(mat).cuda())
        
        h, pool_mask = self.gcn(adj, (to_var(words), to_var(masks), to_var(pos), to_var(ner), torch.from_numpy(dep).type(torch.FloatTensor)))
        
        # pooling
        subj_mask, obj_mask = to_var(subj_pos).eq(0).eq(0).unsqueeze(2), to_var(obj_pos).eq(0).eq(0).unsqueeze(2)  # invert mask
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
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + opt['dep_dim']

        self.emb, self.pos_emb, self.ner_emb, self.dep_emb = embeddings

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
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        words, masks, pos, ner, dep = inputs
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        # if self.opt['dep_dim'] > 0:
        #     embs += [self.dep_emb(dep)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
        else:
            gcn_inputs = embs
        
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            if self.opt['dep_dim'] > 0:
                chosen_deps = self.dep_emb(dep)
                chosen_deps = chosen_deps.sum(2)
                Ax = torch.cat([Ax, chosen_deps], dim=2)
                loop = self.dep_emb(torch.zeros_like(chosen_deps))
                gcn_inputs = torch.cat([gcn_inputs, loop], dim=2)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
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


def id2str(iids, vocab):
    vocab_rev = {v: k for k,v in vocab.items()}
    return [vocab_rev[iid] for iid in iids]
