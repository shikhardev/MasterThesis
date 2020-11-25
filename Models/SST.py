import numpy as np
import torch.nn as nn

class WordAveragingLinear(nn.Module):
    def __init__(self, vocab_size, embedding_dims, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, embedding_dims, padding_idx=0)
        self.out = nn.Linear(embedding_dims, num_classes)
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.num_classes = num_classes
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-np.sqrt(6. / (self.vocab_size+1 + self.embedding_dims)),
                                       np.sqrt(6. / (self.vocab_size+1 + self.embedding_dims)))
        self.out.weight.data.normal_(0, 1 / np.sqrt(self.embedding_dims))
        self.out.bias.data.zero_()

    def forward(self, x):
        return self.out(self.embedding(x).mean(1))
