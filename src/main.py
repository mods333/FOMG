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


""" Some global variables """
_loader = Loader(502) # 500 + SOS + EOS
loader = MIDILoader(_loader)

num_update = 10
use_cuda = torch.cuda.is_available()
# Is the tokenizer 1 indexed?
vocabulary_size = 16*128*2 + 32*16 + 100 + 1 # 4708 + 1
vocabulary_size = vocabulary_size + 2 # SOS (index 4709) and EOS (index 4710)
SOS_TOKEN = 4709
EOS_TOKEN = 4710

encoding_size = 500
one_hot_embeddings = np.eye(vocabulary_size)

lr = 0.01
meta_learner = MetaLearner(Model,
(vocabulary_size,encoding_size,vocabulary_size,lr), lr, num_update)


print_every = 10
check_every = 300
total_epochs = 3000
print_loss_total = [0,0,0]
startTime = time.time()

eps = load_sampler_from_config("../src/config.yaml")

for epoch in  range(1,total_epochs+1):
    
    episode = eps.get_episode()
    train = episode.support
    test = episode.query
    
    loss = meta_learner(train, test)

    print_loss_total = loss

    if epoch % print_every == 0:
        #print_loss_avg = print_loss_total / print_every
        print_loss_avg = [x/print_every for x in print_loss_total]
        print_loss_total = [0,0,0] 
        print('%s (%d %d%%) %s' % (timeSince(startTime, epoch / total_epochs),
                                     epoch, epoch / total_epochs * 100,
                                     str(print_loss_avg)))
    if epoch % check_every == 0:
        torch.save(meta_learner.state_dict(), '../models/maml_e-2_'+str(epoch))

print_loss_total = [0,0,0]
startTime = time.time()


lr = 0.001
meta_learner = MetaLearner(Model,
(vocabulary_size,encoding_size,vocabulary_size,lr), lr, num_update)

eps = load_sampler_from_config("../src/config.yaml")


for epoch in range(1,total_epochs+1):
    
    episode = eps.get_episode()
    train = episode.support
    test = episode.query
    
    loss = meta_learner(train, test)

    print_loss_total = loss

    if epoch % print_every == 0:
        #print_loss_avg = print_loss_total / print_every
        print_loss_avg = [x/print_every for x in print_loss_total]   
 
        print_loss_total = [0,0,0] 
        print('%s (%d %d%%) %s' % (timeSince(startTime, epoch / total_epochs),
                                     epoch, epoch / total_epochs * 100,
                                     str(print_loss_avg)))
    if epoch % check_every == 0:
        torch.save(meta_learner.state_dict(), '../models/maml_e-3_'+str(epoch))

print_loss_total = [0,0,0]
startTime = time.time()



lr = 0.0001
meta_learner = MetaLearner(Model,
(vocabulary_size,encoding_size,vocabulary_size,lr), lr, num_update)

eps = load_sampler_from_config("../src/config.yaml")

for epoch in range(1,total_epochs+1):
    
    episode = eps.get_episode()
    train = episode.support
    test = episode.query
    
    loss = meta_learner(train, test)

    print_loss_total = loss

    if epoch % print_every == 0:
        #print_loss_avg = print_loss_total / print_every
        print_loss_avg = [x/print_every for x in print_loss_total]   
 
        print_loss_total = [0,0,0] 
        print('%s (%d %d%%) %s' % (timeSince(startTime, epoch / total_epochs),
                                     epoch, epoch / total_epochs * 100,
                                     str(print_loss_avg)))
    if epoch % check_every == 0:
        torch.save(meta_learner.state_dict(), '../models/maml_e-4_'+str(epoch))

