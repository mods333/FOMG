from common import *
from loaders import Loader, MIDILoader
from model import Model
from meta_learner import MetaLearner

import numpy as np
import torch
from torch.autograd import Variable


""" Init Models """
_loader = Loader(502)  # 500 + SOS + EOS
loader = MIDILoader(_loader)
lr = 0.01
meta_learner = MetaLearner(Model, (vocabulary_size,encoding_size,vocabulary_size,lr), 1, 0.01, 2)

input_files = ['bach_846.mid', 'mz_311_1.mid', 'rac_op3_2.mid']
input_variables = []
original_sequences = []

for index, input_file in enumerate(input_files):
    orig_seq = loader.read('../data/' + input_file)
    orig_seq = loader.tokenize(orig_seq)

    trunc_seq = orig_seq[0:500]
    trunc_seq = [SOS_TOKEN] + trunc_seq + [EOS_TOKEN]
    original_sequences.append(trunc_seq)
    seq_length = len(trunc_seq)

    # This is really time consuming
    trunc_seq = torch.from_numpy(np.array(one_hot_embeddings[trunc_seq]))
    trunc_seq = trunc_seq.view(seq_length, vocabulary_size)
    trunc_seq = Variable(trunc_seq)
    if use_cuda:
        trunc_seq = trunc_seq.cuda()
    input_variables.append(trunc_seq)

original_sequences = np.array(original_sequences, dtype=np.int64)

# (T_i, seq_length, batch_size, vocab_size)
# accs = meta_learner(input_variables[0].view(
#     1, 502, 1, vocabulary_size), input_variables[1].view(1, 502, 1, vocabulary_size), Variable(torch.from_numpy(original_sequences[0])).view(502, 1).cuda())
# print(accs)

print(meta_learner.learner.meta_net.map_inference(
    input_variables[0][0:100].view(100, 1, vocabulary_size)))
