
# coding: utf-8

# In[2]:


from loaders import *
from collections import Counter
from random import random
from torch import nn
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
    
    def __init(self,network_class,*args):
        
        #define the network for the learner and the meta-learner network
        self.meta_net = network_class(*args)
        self.learner_net = network_class(*args)
        
        self.optimizer = optim.SGD(self.learner_net.parameters(), 0.1)
        
    
    def copy_theta(self):
        
        self.learner_net.load_state_dict(self.meta_net.state_dict())
        
    def forward(self, support_x, support_y, query_x, query_y,num_updates):
        
        #Copy theta into theta'
        self.copy_theta()
        
        # update for several steps
        for i in range(num_updates):
            # forward and backward to update net_pi grad.
            loss, pred = self.learner_net(support_x, support_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        #Find the loss on the query set
        loss, pred = self.learner_net(query_x, query_y)
        # pred: [setsz, n_way], indices: [setsz]
        _, indices = torch.max(pred, dim=1)
        correct = torch.eq(indices, query_y).sum().data[0]
        acc = correct / query_y.size(0)

        grads_pi = autograd.grad(loss, self.learner_net.parameters(), create_graph=True)

        return loss, grads_pi, acc

    def net_forward(self, support_x, support_y):
        
        loss, pred = self.net(support_x, support_y)
        return loss, pred              


# In[4]:


class MetaLearner(nn.Module):
    
    def __init(self,network_class,network_args,k_shot,beta,num_updates):
        
        super(MetaLearner, self).__init__()

        self.k_shot = k_shot
        self.beta = beta
        self.num_updates = num_updates

        # it will contains a learner class to learn on episodes and gather the loss together.
        self.learner = Learner(network_class, *net_cls_args)
        # the optimizer is to update theta parameters, not theta_pi parameters.
        self.optimizer = optim.Adam(self.learner.parameters(), lr=beta)
        
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

    def forward(self, support_x, support_y, query_x, query_y):
        
        sum_grads_pi = None
        meta_batchsz = support_y.size(0)

        # support_x[i]: [setsz, c_, h, w]
        # we do different learning task sequentially, not parallel.
        accs = []
        # for each task/episode.
        for i in range(meta_batchsz):
            _, grad_pi, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)
            if sum_grads_pi is None:
                sum_grads_pi = grad_pi
            else:  # accumulate all gradients from different episode learner
                sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi)]

        # As we already have the grads to update
        # We use a dummy forward / backward pass to get the correct grads into self.net
        # the right grads will be updated by hook, ignoring backward.
        # use hook mechnism to write sumed gradient into network.
        # we need to update the theta/net network, we need a op from net network, so we call self.learner.net_forward
        # to get the op from net network, since the loss from self.learner.forward will return loss from net_pi network.
        dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0])
        self.write_grads(dummy_loss, sum_grads_pi)

        return accs
    
    def pred(self, support_x, support_y, query_x, query_y):
        """
        predict for query_x
        :param support_x:
        :param support_y:
        :param query_x:
        :param query_y:
        :return:
        """
        meta_batchsz = support_y.size(0)

        accs = []
        # for each task/episode.
        # the learner will copy parameters from current theta network and then fine-tune on support set.
        for i in range(meta_batchsz):
            _, _, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)

        return np.array(accs).mean()
    

