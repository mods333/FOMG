
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
import random

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

    def forward(self, input, hidden_in):
        # encoder only outputs hidden
        _, hidden_out = self.lstm(input, hidden_in)
        return hidden_out

    def initHidden(self):

        result = Variable(torch.zeros(1, 1, self.hidden_size)).double()

        if use_cuda:
            result = result.cuda()
        return result


# In[3]:


class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size).double()
        self.out = nn.Linear(hidden_size, output_size).double()
        self.project = nn.Linear(4096, self.hidden_size).double()
        if use_cuda:
            self.lstm = self.lstm.cuda()
            self.out = self.out.cuda()
            self.project = self.project.cuda()

    def forward(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        output = output.squeeze()
        return output.unsqueeze(0), hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size)).double()
        if use_cuda:
            return result.cuda()
        else:
            return result


# In[ ]:


# The next two functions are part of some other deep learning frameworks, but PyTorch
# has not yet implemented them. We can find some commonly-used open source worked arounds
# after searching around a bit: https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1.
def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def compute_loss(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.double()
    loss = losses.sum() / length.double().sum()
    return loss


# In[9]:


class Model(nn.Module):

    def __init__(self, vocabulary_size, encoding_size):
        super(Model, self).__init__()
        self.encoder = EncoderLSTM(vocabulary_size, encoding_size)
        self.decoder = DecoderLSTM(
            vocabulary_size, encoding_size, vocabulary_size)
        self.teacher_forcing = 0.9

    def forward(self, sequence, numbered_seq):

        encoder = self.encoder
        decoder = self.decoder

        # (seq_length, batch_size, vocab_size)
        seq_size = sequence.size()
        batch_size = seq_size[1]
        sequence_length = seq_size[0]
        loss = 0

        encoder_hidden = (encoder.initHidden(), encoder.initHidden())

        # Encoder is fed the flipped control sequence
        for index_control in np.arange(sequence_length-1, 0, -1):
            encoder_input = sequence[index_control].unsqueeze(
                0)  # (1, batch_size, vocab_size)
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # feed encoder_hidden
        decoder_input = sequence[1].unsqueeze(0)  # One after SOS
        decoder_hidden = encoder_hidden
        predicted_note_index = 0

        # Prepare the results tensor
        # (seq_length, batch_size, vocab_size)
        all_decoder_outputs = Variable(torch.zeros(*sequence.size())).double()
        if use_cuda:
            all_decoder_outputs = all_decoder_outputs.cuda()

        all_decoder_outputs[0] = decoder_input

        for index_control in range(2, sequence_length):
            # decoder_input = decoder_input.view(1, 1, vocabulary_size)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            if random.random() <= self.teacher_forcing:
                decoder_input = sequence[index_control].unsqueeze(0)
            else:
                topv, topi = decoder_output.data.topk(1)
                # This is the next input, without teacher forcing it's the predicted output
                decoder_input = torch.stack([Variable(torch.DoubleTensor(one_hot_embeddings[ni]))
                                             for ni in topi.squeeze()]).unsqueeze(0)
                if use_cuda:
                    decoder_input = decoder_input.cuda()

            # Save the decoder output
            all_decoder_outputs[index_control] = decoder_output

        seq_lens = Variable(torch.LongTensor(
            np.ones(batch_size, dtype=int)*sequence_length))
        if use_cuda:
            seq_lens = seq_lens.cuda()

        loss = compute_loss(all_decoder_outputs.transpose(0, 1).contiguous(),
                            numbered_seq.transpose(0, 1).contiguous(),
                            seq_lens)

        return loss

    def map_inference(self, sequence, embeddings=one_hot_embeddings, max_length=500):
        """ sequence has to be batch_size=1"""
        encoder = self.encoder
        decoder = self.decoder

        output_control_sequence = []

        # Encoder
        encoder_hidden = self.hidden

        sequence_length = sequence.size()[1]

        for index_control in np.arange(sequence_length-1, 0, -1):
            encoder_input = sequence[0][index_control].view(
                1, 1, vocabulary_size)
            # Gets hidden for next input
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # This point we have last encoder_hidden, feed into decoder
        decoder_hidden = encoder_hidden
        decoder_input = sequence[0][0]
        predicted_control_index = SOS_TOKEN

        cur_length = 0
        while True:
            decoder_input = decoder_input.view(1, 1, vocabulary_size)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            # MAP inference
            topv, topi = decoder_output.data.topk(1)
            predicted_control_index = int(topi)
            if predicted_control_index == EOS_TOKEN:
                break
            output_control_sequence.append(predicted_control_index)

            # This is the next input
            decoder_input = torch.from_numpy(
                embeddings[predicted_control_index])
            decoder_input = Variable(decoder_input).double()
            if use_cuda:
                decoder_input = decoder_input.cuda()

            cur_length += 1
            if cur_length >= max_length:
                break

        return output_control_sequence
