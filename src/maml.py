import numpy as np
import random
import torch

import torch.nn.functional as F
from torch import nn
from torch import autograd
from torch.autograd import Variable


import sys
sys.path.insert(0, '../src')
from episode import *


# ### Global Variables
use_cuda = torch.cuda.is_available()
vocabulary_size = 16*128*2 + 32*16 + 100 + 1  # 4708 + 1
vocabulary_size = vocabulary_size + 2  # SOS (index 4709) and EOS (index 4710)
SOS_TOKEN = 4709
EOS_TOKEN = 4710

encoding_size = 500
one_hot_embeddings = np.eye(vocabulary_size)


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


# ### Learner and MetaLearner Network

class Learner(nn.Module):

    def __init__(self, network_class, *args):

        super(Learner, self).__init__()
        # define the network for the learner and the meta-learner network
        self.meta_net = network_class(*args)
        self.learner_net = network_class(*args)

        self.optimizer = torch.optim.SGD(self.learner_net.parameters(), 0.1)

    def copy_theta(self):

        # Ablation test -- set to 0s
        self.learner_net.load_state_dict(self.meta_net.state_dict())

    def forward(self, support_x, query_x, num_updates, support_cat, query_cat):

        # Copy theta into theta'
        self.copy_theta()

        # update for several steps
        for i in range(num_updates):
            # forward and backward to update net_pi grad.
            loss = self.learner_net(support_x, support_cat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Find the loss on the query set
        loss = self.learner_net(query_x, query_cat)

        grads_pi = autograd.grad(
            loss, self.learner_net.parameters(), retain_graph=True)

        return loss, grads_pi, loss.data[0]

    def net_forward(self, support_x, numbered_seq):

        loss = self.meta_net(support_x, numbered_seq)
        return loss

class MetaLearner(nn.Module):

    def __init__(self, network_class, network_args, k_shot, beta, num_updates):

        super(MetaLearner, self).__init__()

        self.k_shot = k_shot
        self.beta = beta
        self.num_updates = num_updates

        # it will contains a learner class to learn on episodes and gather the loss together.
        self.learner = Learner(network_class, *network_args)
        # the optimizer is to update theta parameters, not theta_pi parameters.
        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=beta)

    def write_grads(self, dummy_loss, sum_grads_pi):
        """
        write loss into learner.net, gradients come from sum_grads_pi.
        Since the gradients info is not calculated by general backward, we need this function to write the right gradients
        into theta network and update theta parameters as wished.
        :param dummy_loss: dummy loss, nothing but to write our gradients by hook
        :param sum_grads_pi: the summed gradients
        :return:
        """

        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []

        for i, v in enumerate(self.learner.parameters()):
            def closure():
                ii = i
                return lambda grad: sum_grads_pi[ii]

            # if you write: hooks.append( v.register_hook(lambda grad : sum_grads_pi[i]) )
            # it will pop an ERROR, i don't know why?
            hooks.append(v.register_hook(closure()))

        # use our sumed gradients_pi to update the theta/net network,
        # since our optimizer receive the self.net.parameters() only.
        self.optimizer.zero_grad()
        dummy_loss.backward()
        self.optimizer.step()

        # if you do NOT remove the hook, the GPU memory will expode!!!
        for h in hooks:
            h.remove()

    def forward(self, support_x, query_x, support_cat, query_cat):

        sum_grads_pi = None
        # (T_i, seq_length, batch_size, vocab_size)
        meta_batchsz = support_x.size(0)

        # we do different learning task sequentially, not parallel.
        accs = []
        # for each task/episode.
        for i in range(meta_batchsz):
            # ASSUME QUERY SET IS ALWAYS SIZE 1
            _, grad_pi, episode_acc = self.learner(
                support_x[i], query_x[0], self.num_updates, support_cat[i], query_cat[0])
            accs.append(episode_acc)
            if sum_grads_pi is None:
                sum_grads_pi = grad_pi
            else:  # accumulate all gradients from different episode learner
                sum_grads_pi = [torch.add(i, j)
                                for i, j in zip(sum_grads_pi, grad_pi)]

        # As we already have the grads to update
        # We use a dummy forward / backward pass to get the correct grads into self.net
        # the right grads will be updated by hook, ignoring backward.
        # use hook mechnism to write sumed gradient into network.
        # we need to update the theta/net network, we need a op from net network, so we call self.learner.net_forward
        # to get the op from net network, since the loss from self.learner.forward will return loss from net_pi network.
        dummy_loss = self.learner.net_forward(support_x[0], support_cat[0])
        self.write_grads(dummy_loss, sum_grads_pi)

        return accs


# ### Model Architecture
# Just a simple LSTM encoder and decoder

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

    def initHidden(self, batch_size):

        result = Variable(torch.zeros(
            1, batch_size, self.hidden_size)).double()

        if use_cuda:
            result = result.cuda()
        return result

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
        # output = F.relu(input) 
        # FIXME: Don't think we need the RELU?  Input is one-hot, RELU does nothing. 
        output, hidden = self.lstm(input, hidden)
        output = self.out(output)
        output = output.squeeze()
        return output.unsqueeze(0), hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(
            1, batch_size, self.hidden_size)).double()
        if use_cuda:
            return result.cuda()
        else:
            return result


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

        encoder_hidden = (encoder.initHidden(batch_size),
                          encoder.initHidden(batch_size))

        # Encoder is fed the flipped control sequence
        for index_control in np.arange(sequence_length-1, 0, -1):
            encoder_input = sequence[index_control].unsqueeze(
                0)  # (1, batch_size, vocab_size)
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # feed encoder_hidden
        decoder_input = sequence[0].unsqueeze(0)  # This is SOS
        decoder_hidden = encoder_hidden

        # Prepare the results tensor
        # (seq_length, batch_size, vocab_size)
        all_decoder_outputs = Variable(torch.zeros(*sequence.size())).double()
        if use_cuda:
            all_decoder_outputs = all_decoder_outputs.cuda()

        all_decoder_outputs[0] = decoder_input

        for index_control in range(1, sequence_length):
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
        """
        Input:
            sequence: (seq_length, batch, vocab_size)
            output: [[seq1], ..., [seqN]] where N is number of batch
        """
        encoder = self.encoder
        decoder = self.decoder

        # (seq_length, batch_size, vocab_size)
        seq_size = sequence.size()
        batch_size = seq_size[1]
        sequence_length = seq_size[0]

        encoder_hidden = (encoder.initHidden(batch_size),
                          encoder.initHidden(batch_size))

        # Encoder is fed the flipped control sequence
        for index_control in np.arange(sequence_length-1, 0, -1):
            encoder_input = sequence[index_control].unsqueeze(
                0)  # (1, batch_size, vocab_size)
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # feed encoder_hidden
        decoder_input = sequence[0].unsqueeze(0)  # This is SOS
        decoder_hidden = encoder_hidden

        output_control_sequences = [[] for batch in range(batch_size)]
        append_flag = [True for batch in range(batch_size)]
        # Prepare the results tensor
        # (seq_length, batch_size, vocab_size)
        index_control = 1
        while True:
            # decoder_input = decoder_input.view(1, 1, vocabulary_size)
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)

            next_input = []
            for index, ni in enumerate(topi.squeeze()):
                next_input.append(
                    Variable(torch.DoubleTensor(one_hot_embeddings[ni])))
                # If we hit an EOS, stop appending to that output sequence
                if ni == EOS_TOKEN:
                    append_flag[index] = False
                if append_flag[index]:
                    output_control_sequences[index].append(ni)

            decoder_input = torch.stack(next_input).unsqueeze(0)

            if use_cuda:
                decoder_input = decoder_input.cuda()

            index_control += 1
            if index_control >= max_length:
                break

        return output_control_sequences


# ### Training

# In[10]:


base_path = '../data/small/'
training_set = np.load(base_path + 'beethoven_brunomars_eminem_mozart.npy')
train_size = len(training_set)


# In[117]:


""" RUN THIS ONLY MANUALLY! """
meta_learner = MetaLearner(Model, (vocabulary_size, encoding_size), 1, 0.01, 2)


# In[50]:


""" Convert to one-hot """
SEQ_LENGTH = 250
training_one_hot = []
training_categories = []
for i in range(train_size):
    training_categories.append(np.concatenate(([SOS_TOKEN], training_set[i][0:SEQ_LENGTH], [EOS_TOKEN])))
    training_one_hot.append(one_hot_embeddings[training_categories[i]])
    
training_one_hot = np.array(training_one_hot)
training_categories = np.array(training_categories, dtype=np.int64)


total_epochs = 10
TASK_BATCH_SIZE = 4 + 1 # 3 support + 1 query
SEQ_LENGTH += 2 # SOS and EOS

print_every = 1
check_every = 1
print_loss_total = 0
startTime = time.time()
randomize_song_index = np.arange(train_size)
for epoch in range(1, total_epochs+1):
    # Randomize
    np.random.shuffle(randomize_song_index)
    training_one_hot = training_one_hot[randomize_song_index]
    training_categories = training_categories[randomize_song_index]
    
    for task_batch in range(train_size // TASK_BATCH_SIZE):
        batch_start = task_batch * TASK_BATCH_SIZE
        batch_end = batch_start + TASK_BATCH_SIZE
        batch = training_one_hot[batch_start:batch_end]
        
        support = batch[:-1]
        support_cat = training_categories[batch_start:batch_end-1]
        query = batch[-1:]
        query_cat = training_categories[batch_end-1:batch_end]
        
        support = Variable(torch.from_numpy(support)).view(TASK_BATCH_SIZE-1, SEQ_LENGTH, 1, vocabulary_size)
        support_cat = Variable(torch.from_numpy(support_cat)).view(TASK_BATCH_SIZE-1, SEQ_LENGTH, 1)
        query = Variable(torch.from_numpy(query)).view(1, SEQ_LENGTH, 1, vocabulary_size)
        query_cat = Variable(torch.from_numpy(query_cat)).view(1, SEQ_LENGTH, 1)
        if use_cuda:
            support = support.cuda()
            support_cat = support_cat.cuda()
            query = query.cuda()
            query_cat = query_cat.cuda()
        # support_x : (T_i, seq_length, batch_size, vocab_size)
        # query_x : (T_i, seq_length, batch_size, vocab_size)
        # num_seq : (T_i, seq_length, batch_size)
        
        loss = meta_learner(support, query, support_cat, query_cat)
        print_loss_total += np.sum(np.array(loss))
        
    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timeSince(startTime, epoch / total_epochs),
                                     epoch, epoch / total_epochs * 100, print_loss_avg)) 
        
    if epoch % check_every == 0:
        torch.save(meta_learner.state_dict(), '../models/maml_'+str(epoch))
        