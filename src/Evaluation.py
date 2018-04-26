
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '../src')

import pickle
from loaders import *
from episode import *
from dataset import *

from common import *
from model import Model
from meta_learner import MetaLearner

import numpy as np
import torch
from torch.autograd import Variable

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[2]:


eps = load_sampler_from_config("../src/config.yaml")


# # Load Models

# In[44]:


num_update = 10
use_cuda = torch.cuda.is_available()
# Is the tokenizer 1 indexed?
vocabulary_size = 16*128*2 + 32*16 + 100 + 1 # 4708 + 1
vocabulary_size = vocabulary_size + 2 # SOS (index 4709) and EOS (index 4710)
SOS_TOKEN = 4709
EOS_TOKEN = 4710

_loader = Loader(502) # 500 + SOS + EOS
loader = MIDILoader(_loader)

encoding_size = 500
one_hot_embeddings = np.eye(vocabulary_size)

lr = 0.001
baseline = Model(vocabulary_size, 
              encoding_size, 
              vocabulary_size,
              learning_rate=lr)
baseline.load_state_dict(torch.load('../models/baseline_e-3_9000'))

meta_learner = MetaLearner(Model,(vocabulary_size,encoding_size,vocabulary_size,lr), lr, num_update)
meta_learner.load_state_dict(torch.load('../models/maml_e-3_3000'))


# In[42]:


from common import *

import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
import random
import torch.utils.data.sampler as sampler

# In[2]:

use_cuda = True
vocabulary_size = 16*128*2 + 32*16 + 100 + 1 + 2  # 4708 + 1
one_hot_embeddings = np.eye(vocabulary_size)

class EncoderLSTM(nn.Module):
    # Your code goes here
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size).double()

        if use_cuda:
            self.lstm = self.lstm.cuda()

    def forward(self, input, hidden):
        _, hidden_out = self.lstm(input, hidden) # encoder only outputs hidden
        return hidden_out

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# In[3]:
class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size).double()
        self.out = nn.Linear(hidden_size, output_size).double()

        if use_cuda:
            self.lstm = self.lstm.cuda()
            self.out = self.out.cuda()

    def forward(self, input, hidden):
        output = F.relu(input)
        #output = input
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        return output[0], hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


# In[9]:

class Model(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 learning_rate,
                 embeddings=one_hot_embeddings):
        super(Model,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = EncoderLSTM(input_size, hidden_size)
        self.decoder = DecoderLSTM(input_size, hidden_size, output_size)
        #self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        #self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)

        self.embeddings = embeddings
        self.criterion = nn.CrossEntropyLoss(reduce=False)

        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.criterion = self.criterion.cuda()

    def forward(self, token_seqs):
        loss = 0
        start = 50
        batch_size = len(token_seqs)
        #seq_len = len(token_seqs[0])
        seq_len = 150
        #self.encoder_optimizer.zero_grad()
        #self.decoder_optimizer.zero_grad()

        encoder_hidden = Variable(self.encoder.initHidden(batch_size)).double()
        encoder_output = Variable(self.encoder.initHidden(batch_size)).double()
        if use_cuda:
            encoder_hidden = encoder_hidden.cuda()
            encoder_output = encoder_output.cuda()

        hidden = (encoder_hidden, encoder_output)
        for i in np.arange(start-1, -1, -1):
            token_batch = np.array(self.embeddings[token_seqs[:, i]])
            encoder_input = Variable(torch.from_numpy(token_batch)).view(1, batch_size, -1).double()
            encoder_input = encoder_input.cuda() if use_cuda else encoder_input
            #print("encoder_input: %d" % (np.where(encoder_input.data==1)[2][0]))
            hidden = self.encoder(encoder_input, hidden)
        encoder_hidden, _ = hidden

        token_batch = np.array(self.embeddings[token_seqs[:, start-1]])
        decoder_input = Variable(torch.from_numpy(token_batch)).double()
        decoder_output = Variable(self.decoder.initHidden(batch_size)).double()
        if use_cuda:
            decoder_output = decoder_output.cuda()

        hidden = (encoder_hidden, decoder_output)
        for i in range(start, seq_len+1):
            decoder_input = decoder_input.squeeze().view(1, batch_size, -1)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            #print("decoder_input: %d" % (np.where(decoder_input.data==1)[2][0]))
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            #print("prediction: %d" % (int(decoder_output.topk(1)[1])))
            if i < seq_len:
                seq_var = token_seqs[:, i]
            else:
                seq_var = [EOS_TOKEN]*batch_size

            target = Variable(torch.from_numpy(np.array(seq_var))).long()
            target = target.cuda() if use_cuda else target
            loss += self.criterion(decoder_output, target)

            # Teacher forcing
            decoder_input = Variable(torch.from_numpy(np.array(self.embeddings[seq_var]))).double()

        loss = torch.sum(loss)/batch_size
        #loss.backward()
        #self.encoder_optimizer.step()
        #self.decoder_optimizer.step()

        return loss

    def map_inference(self, token_seqs):
        batch_size = len(token_seqs)
        seq_len = len(token_seqs[0])
        encoder_hidden = Variable(self.encoder.initHidden(batch_size)).double()
        encoder_output = Variable(self.encoder.initHidden(batch_size)).double()
        if use_cuda:
            encoder_hidden = encoder_hidden.cuda()
            encoder_output = encoder_output.cuda()

        hidden = (encoder_output, encoder_hidden)
        for i in np.arange(seq_len-1, 0, -1):
            token_batch = np.array(self.embeddings[token_seqs[:, i]])
            encoder_input = Variable(torch.from_numpy(token_batch)).view(1, batch_size, -1).double()
            encoder_input = encoder_input.cuda() if use_cuda else encoder_input
            hidden = self.encoder(encoder_input, hidden)

        encoder_output, encoder_hidden = hidden

        token_batch = np.array(self.embeddings[[SOS_TOKEN]*batch_size])
        decoder_output = Variable(self.decoder.initHidden(batch_size)).double()
        if use_cuda:
            decoder_output = decoder_output.cuda()

        hidden = (decoder_output, encoder_hidden)

        pred_seqs = None
        for i in range(150):
            decoder_input = Variable(torch.from_numpy(token_batch)).double()
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            decoder_input = decoder_input.squeeze().view(1, batch_size, -1)
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            topv, topi = decoder_output.data.topk(1)
            #print("Iteration: %d, Prediction: %d" % (i, token))
            if pred_seqs is None:
                pred_seqs = topi.cpu().numpy()
            else:
                pred_seqs = np.concatenate((pred_seqs, topi.cpu().numpy()), axis=1)
            token_batch = np.array(self.embeddings[topi])

        return pred_seqs.tolist()
        
    def sample_inference(self, token_seqs):
        softmax = nn.Softmax()
        batch_size = len(token_seqs)
        seq_len = len(token_seqs[0])
        encoder_hidden = Variable(self.encoder.initHidden(batch_size)).double()
        encoder_output = Variable(self.encoder.initHidden(batch_size)).double()
        if use_cuda:
            encoder_hidden = encoder_hidden.cuda()
            encoder_output = encoder_output.cuda()
        
        hidden = (encoder_output, encoder_hidden)
        for i in np.arange(seq_len-1, -1, -1):
            token_batch = np.array(self.embeddings[token_seqs[:, i]])
            encoder_input = Variable(torch.from_numpy(token_batch)).view(1, batch_size, -1).double()
            encoder_input = encoder_input.cuda() if use_cuda else encoder_input
            hidden = self.encoder(encoder_input, hidden)
        
        encoder_output, encoder_hidden = hidden
            
        token_batch = np.array(self.embeddings[[SOS_TOKEN]*batch_size])
        decoder_output = Variable(self.decoder.initHidden(batch_size)).double()
        if use_cuda:
            decoder_output = decoder_output.cuda()
        
        hidden = (decoder_output, encoder_hidden)

        pred_seqs = None
        for i in range(250):
            decoder_input = Variable(torch.from_numpy(token_batch)).double()
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            decoder_input = decoder_input.squeeze().view(1, batch_size, -1)
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            
            output = softmax(decoder_output).data.cpu()
            output = output.numpy()

            cdf = np.cumsum(output)
            uniform_sample = np.random.uniform()
            
            for _index, item in enumerate(cdf):
                if uniform_sample > item and uniform_sample <= cdf[_index+1]:
                    ni = np.array([[_index]])
                    break

            if ni != EOS_TOKEN and ni != SOS_TOKEN:    
                if pred_seqs is None:
                    pred_seqs = ni
                else:
                    pred_seqs = np.concatenate((pred_seqs, ni), axis=1)
                    
            token_batch = np.array(self.embeddings[ni])
            
        return pred_seqs.tolist()


# In[48]:

import copy

test_model = Model(vocabulary_size, encoding_size, vocabulary_size, learning_rate=lr)
optimizer = torch.optim.Adam(test_model.parameters(), lr)

N = 10 # 30 Artists
num_updates = 100
feed_length = 50
for index in range(N):
    episode = eps.get_episode()
    support = episode.support
    query = episode.query

    for artist_index in range(support.shape[0]):
        """ Baseline """
        test_model.load_state_dict(copy.deepcopy(baseline.state_dict()))
        test_model.train()
        for _ in range(num_updates):
            # train for 10 iterations
            optimizer.zero_grad()
            loss = test_model(support[artist_index])
            loss.backward()
            optimizer.step()
        test_model.eval()
        for inf_index in range(1, query[artist_index].shape[0]+1):
            try:
                # Make inference for each song
                song = query[artist_index][inf_index]
                midi = loader.detokenize(np.array(song))
                midi.write('baseline_{}_{}_orig.mid'.format(index*3+artist_index+1, inf_index))

                midi = loader.detokenize(np.array(song[:feed_length]))
                midi.write('baseline_{}_{}_inpu.mid'.format(index*3+artist_index+1, inf_index))

                gen_seq = test_model.sample_inference(np.array([song[:feed_length]]))
                midi = loader.detokenize(np.append(song[:feed_length], np.array(gen_seq[0])))
                midi.write('baseline_{}_{}_pred.mid'.format(index*3+artist_index+1, inf_index))
            except:
                print('Failed to generate: baseline_{}_{}'.format(index*3+artist_index+1, inf_index))
        """ MAML """
        test_model.load_state_dict(copy.deepcopy(meta_learner.learner.meta_net.state_dict()))
        test_model.train()
        for _ in range(num_updates):
            # train for 10 iterations
            optimizer.zero_grad()
            loss = test_model(support[artist_index])
            loss.backward()
            optimizer.step()
        test_model.eval()
        for inf_index in range(1, query[artist_index].shape[0]+1):
            try:
                # Make inference for each song
                song = query[artist_index][inf_index]
                midi = loader.detokenize(np.array(song))
                midi.write('maml_{}_{}_orig.mid'.format(index*3+artist_index+1, inf_index))

                midi = loader.detokenize(np.array(song[:feed_length]))
                midi.write('maml_{}_{}_inpu.mid'.format(index*3+artist_index+1, inf_index))

                gen_seq = test_model.sample_inference(np.array([song[:feed_length]]))
                midi = loader.detokenize(np.append(song[:feed_length], np.array(gen_seq[0])))
                midi.write('maml_{}_{}_pred.mid'.format(index*3+artist_index+1, inf_index))
            except:
                print('Failed to generate: maml_{}_{}'.format(index*3+artist_index+1, inf_index))

