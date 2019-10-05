"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import pickle

from utils import constant, helper, vocab

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID

        with open(filename) as infile:
            data = json.load(infile)
        with open(opt["data_dir"] + "/%s.pkl" % (filename.split("/")[-1].split(".")[0]), "rb") as infile:
            sents = pickle.load(infile)
        data = self.preprocess(data, vocab, opt, sents)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt, sents):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d, sent in zip(data, sents):
            # tokens = list(d['token'])
            tokens = []
            sent_vals = list(sent.values())
            for t in sent_vals:
                tokens.append(t.get_conllu_field("form"))
                # TODO - dep
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            for i in range(len(sent) - len(d['token'])):
                if sent_vals[len(d['token']) + i].get_conllu_field("misc").startswith("CopyOf="):
                    copy_i = int(sent_vals[len(d['token']) + i].get_conllu_field("id")) - 1
                    pos.append(pos[copy_i])
                    ner.append(ner[copy_i])
                else:
                    pos.append(constant.UNK_ID)
                    ner.append(constant.UNK_ID)

            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            # assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]

            dep = []
            if self.opt["dep_dim"]:
                if self.opt["dep_type"] in [constant.DepType.NAKED.value, constant.DepType.SPLITED.value]:
                    dep = [[constant.DEP_TO_ID2[":".join(r.split(":")[:2]) if ":".join(r.split(":")[:2]) in constant.DEP_TO_ID2 else r.split(":")[0]] for (c, r) in t.get_children_with_rels()] for t in sent_vals]
                    if self.opt["dep_type"] == constant.DepType.SPLITED.value:
                        dep2 = [[constant.DEP_CASE_INFO[r.split(":")[1]] for (c, r) in t.get_children_with_rels() if (len(r.split(":")) > 1) and (r.split(":")[1] in constant.DEP_CASE_INFO)] for t in sent_vals]
                        dep3 = [[constant.DEP_EXTRA[r.split(":")[-1].split("_")[0]] for (c, r) in t.get_children_with_rels() if "_extra" in r.split(":")[-1]] for t in sent_vals]
                        dep = (dep, dep2, dep3)
                else:  # self.opt["dep_type"] == constant.DepType.ALL.value
                    dep = [[constant.DEP_TO_ID[r] for (c, r) in t.get_children_with_rels()] for t in sent_vals]
            
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, sent_vals, dep, relation)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 12

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[11])
        sents = batch[9]
        dep = batch[10]
        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx, sents, dep)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

