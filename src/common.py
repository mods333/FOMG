""" Some global variables """
import numpy as np
import torch

use_cuda = torch.cuda.is_available()
# Is the tokenizer 1 indexed?
vocabulary_size = 16*128*2 + 32*16 + 100 + 1  # 4708 + 1
vocabulary_size = vocabulary_size + 2  # SOS (index 4709) and EOS (index 4710)
SOS_TOKEN = 4709
EOS_TOKEN = 4710

encoding_size = 500
one_hot_embeddings = np.eye(vocabulary_size)
