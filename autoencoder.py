import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


# We probably want a sequence to sequence autoencoder here really.
# we want to build an auto encoder that could represent questions and answers
# So given a series of words of a question- can we output a series of words of a question
# what about if we want to use different autoencoders for the questions vs the answers?
# do we even care about answers in this preliminary version
# (I don't think so- we only care about being able to learn a representation of a question right?)
# base line: 
# need to duplicate questions -> [question,context], [question,dud_context] in training
# in eval -> [question, dud_context] -> answer
# we can have a hidden state that's huge
# and then some mid vector it needs to learn to look up that data
# 
class Encoder(nn.Module):
    def __init__(self,embed_size,encoding_size):
        super(Encoder, self).__init__()
        
        self.flinear = nn.Linear(embed_size, embed_size)
        self.slinear =  nn.Linear(embed_size, encoding_size)

    def forward(self, x):
        return F.sigmoid(self.slinear(F.sigmoid(self.flinear(x))))

class Decoder(nn.Module):
    def __init__(self,embed_size,encoding_size):
        super(Decoder, self).__init__()
        self.flinear = nn.Linear(encoding_size, encoding_size)
        self.slinear = nn.Linear(encoding_size, embed_size)
        
    def forward(self, x):
        return F.sigmoid(self.slinear(F.sigmoid(self.flinear(x))))

class AutoEncoder(nn.Module):
    def __init__(self, embed_size,encoding_size):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder(embed_size, encoding_size)
        self.dec = Decoder(embed_size, encoding_size)

    def forward(self, x):
        return self.dec(self.enc(x))