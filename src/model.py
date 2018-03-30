
# coding: utf-8

# In[1]:

from common import *

import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
import random

# In[2]:

use_cuda = False
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
        for i in np.arange(start-1, 0, -1):
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
        for i in range(500):
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
