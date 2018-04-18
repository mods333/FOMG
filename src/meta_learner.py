
# coding: utf-8

# In[2]:


from loaders import *
from collections import Counter
from random import random
from torch import nn
from torch import autograd
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import random


# In[3]:


class Learner(nn.Module):

    def __init__(self, network_class, lr,*args):

        super(Learner, self).__init__()
        # define the network for the learner and the meta-learner network
        self.meta_net = network_class(*args)
        self.learner_net = network_class(*args)

        self.optimizer = torch.optim.Adam(self.learner_net.parameters(), lr)

    def copy_theta(self):

        # Ablation test -- set to 0s
        self.learner_net.load_state_dict(self.meta_net.state_dict())

    def forward(self, support_x, query_x, num_updates):

        # Copy theta into theta'
        self.copy_theta()
        seq_len = 150
        
        # update for several steps
        for i in range(num_updates):
            # forward and backward to update net_pi grad.
            loss = self.learner_net(support_x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Find the loss on the query set
        loss = self.learner_net(query_x)

        grads_pi = autograd.grad(
            loss, self.learner_net.parameters(), retain_graph=True, allow_unused=True)

        return loss, grads_pi, loss.data[0]/seq_len

    def net_forward(self, support_x):

        loss = self.meta_net(support_x)
        return loss


# In[4]:


class MetaLearner(nn.Module):

    def __init__(self, network_class, network_args, beta, num_updates):

        super(MetaLearner, self).__init__()

        self.beta = beta
        self.num_updates = num_updates

        # it will contains a learner class to learn on episodes and gather the loss together.
        self.learner = Learner(network_class,beta, *network_args,)
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

    def forward(self, support_x, query_x):

        sum_grads_pi = None
        # (T_i, seq_length, batch_size, vocab_size)
        
        meta_batchsz = len(support_x)

        # we do different learning task sequentially, not parallel.
        accs = []
        # for each task/episode.
        for i in range(meta_batchsz):
            _, grad_pi, episode_acc = self.learner(support_x[i], query_x[i], self.num_updates)
            
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
        dummy_loss = self.learner.net_forward(support_x[0])
        self.write_grads(dummy_loss, sum_grads_pi)

        return accs
