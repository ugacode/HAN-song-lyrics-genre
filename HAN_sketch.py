import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

"""
from .embed_regularize import embedded_dropout
from .locked_dropout import LockedDropout
from .weight_dropout import WeightDrop
"""

"""
blablablubb

Take word attention network, go through embeddings and put resulting token embeddings
in the bidirectional GRU (rnn). Then put the output of the GRU to the AttentionWithContext-Class (word_attn)
where you get the importance weights (α), sentence representation (s) and the hidden state h_n

Returning the hidden state is necessary since the document (amazon review, lyrics or what so ever)
 is comprised of a series of sentences.
 --> initial hidden state of sentence i+1 will be the last hidden state of sentence i. 
 ... it's kind of like treating the documents themselves as "stateful"

"""






class AttentionWithContext(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionWithContext, self).__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)

                                            # "inp" refers to h_it and h_i for word and sentence attention respectively.
    def forward(self, inp):
        u = torch.tanh_(self.attn(inp))     # hidden representation of h_it, u_i
        a = F.softmax(self.contx(u), dim=1) # measure importance of each word; obtaining normalised importance weight α_it.
        s = (a * inp).sum(1)                # s = sentence vector; it is weighted sum of the h_it features based on normalised importance weights
        return a.permute(0, 2, 1), s


                                            # word encoder + word attention
class WordAttnNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=32,
        padding_idx=1,
        embed_dim=50,
        weight_drop=0.0,
        embed_drop=0.0,
        locked_drop=0.0,
        embedding_matrix=None,
    ):
        super(WordAttnNet, self).__init__()

        self.lockdrop = LockedDropout()
        self.embed_drop = embed_drop
        self.weight_drop = weight_drop
        self.locked_drop = locked_drop

        if isinstance(embedding_matrix, np.ndarray):
            self.word_embed = nn.Embedding(
                vocab_size, embedding_matrix.shape[1], padding_idx=padding_idx
            )
            self.word_embed.weight = nn.Parameter(torch.Tensor(embedding_matrix))
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        if weight_drop:
            self.rnn = WeightDrop(
                self.rnn, ["weight_hh_l0", "weight_hh_l0_reverse"], dropout=weight_drop
            )

        self.word_attn = AttentionWithContext(hidden_dim * 2)

    def forward(self, X, h_n):
        if self.embed_drop:
            embed = embedded_dropout(
                self.word_embed, X.long(), dropout=self.embed_drop if self.training else 0,
            )
        else:
            embed = self.word_embed(X.long())           # input tokens (X) go through embeddings lookup table
        if self.locked_drop:
            embed = self.lockdrop(embed, self.locked_drop)

        h_t, h_n = self.rnn(embed, h_n)
        a, s = self.word_attn(h_t)
        return a, s.unsqueeze(1), h_n




                                            # for sentence encoder + sentence attention we do not an need embedding lookup table
class SentAttnNet(nn.Module):
    def __init__(self, word_hidden_dim=32, sent_hidden_dim=32, padding_idx=1,
        weight_drop=0.0):
        super(SentAttnNet, self).__init__()

        self.rnn = nn.GRU( word_hidden_dim * 2, sent_hidden_dim, bidirectional=True,
            batch_first=True)
        if weight_drop:
            self.rnn = WeightDrop(self.rnn, ["weight_hh_l0", "weight_hh_l0_reverse"],
                dropout=weight_drop)
        self.sent_attn = AttentionWithContext(sent_hidden_dim * 2)

    def forward(self, X):
        h_t, h_n = self.rnn(X)
        a, v = self.sent_attn(h_t)
        return a, v

#Let’s assume we use batch sizes ( bsz ) of 32, token embedding of dim ( embed_dim ) 100 and GRUs with hidden size ( hidden_dim ) 64.
class HierAttnNet(nn.Module):

    def __init__( self, vocab_size, maxlen_sent, maxlen_doc, word_hidden_dim=32,
        sent_hidden_dim=32, padding_idx=1, embed_dim=50, weight_drop=0.0, embed_drop=0.0,
        locked_drop=0.0, last_drop=0.0, embedding_matrix=None, num_class=4):
        super(HierAttnNet, self).__init__()

        self.word_hidden_dim = word_hidden_dim
        self.wordattnnet = WordAttnNet(vocab_size, word_hidden_dim, padding_idx, embed_dim,
            weight_drop, embed_drop, locked_drop, embedding_matrix)
        self.sentattnnet = SentAttnNet(word_hidden_dim, sent_hidden_dim, padding_idx,
            weight_drop)
        self.ld = nn.Dropout(p=last_drop)
        self.fc = nn.Linear(sent_hidden_dim * 2, num_class)

    def forward(self, X):
        x = X.permute(1, 0, 2)
        word_h_n = nn.init.zeros_(torch.Tensor(2, X.shape[0], self.word_hidden_dim))
        if use_cuda:
            word_h_n = word_h_n.cuda()
        word_a_list, word_s_list = [], []
        for sent in x:
            word_a, word_s, word_h_n = self.wordattnnet(sent, word_h_n)
            word_a_list.append(word_a)
            word_s_list.append(word_s)
        self.sent_a = torch.cat(word_a_list, 1)
        sent_s = torch.cat(word_s_list, 1)
        doc_a, doc_s = self.sentattnnet(sent_s)
        self.doc_a = doc_a.permute(0, 2, 1)
        doc_s = self.ld(doc_s)
        return self.fc(doc_s)



print("hi world")