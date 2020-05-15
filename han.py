import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
from word_encoding import WordEncodingAuto

MAX_WORDS_PER_VERSE = 10


def matrix_mul(out_features, weight, bias=False):
    feature_list = []
    for feature in out_features:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return feature_list


def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


def embedder(verse, emb_size):
    emb_verse = []
    embed_dim = 100
    WE = WordEncodingAuto('6B', embed_dim)

    for word in verse.split(' '):
        emb_word = WE.get_word_vector(word).reshape(1, -1)
        emb_verse.append(emb_word)

    # 6x1x100 dont know if stacking is the right thing because it adds one dimension
    # do padding and truncating here
    tensor_filler = torch.Tensor(1, 100)
    emb_verse = torch.stack(emb_verse)

    return emb_verse


class WordAttNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        print('starting init')
        super(WordAttNet, self).__init__()
        # dict = pd.read_csv(filepath_or_buffer=DATASET_PATH, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]

        # get the data from csv
        # dict = pd.read_csv(filepath_or_buffer='testcsv.csv', header=None, sep=",").values[1:, 1:]

        self.word_weight = nn.Parameter(
            torch.Tensor(2 * hidden_dim, 2 * hidden_dim))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_dim))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_dim, 1))

        self.attn = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.contx = nn.Linear(hidden_dim * 2, 1)

        # embed_dim number of features in the input X
        # hidden_dim number of features in the hidden state h
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True)

        # create weight and contex
        self.word_weight.data.normal_(0.0, 0.5)
        self.context_weight.data.normal_(0.0, 0.5)

        print('finished init')

    def forward(self, my_input, hidden_state):
        print('starting forward')

        # output is of shape(seq_len x batch x input_size)
        # apply the gru to obtain the tensor of the output features (seq_len x batch x num_directions * hidden_dim)
        # and the hidden state h_n (num_layers * num_directions x batch x hidden_dim)
        # num_directions = 2 because the RNN is bidirectional
        #print('output shape is: {}'.format(output.shape))
        out_features, h_n = self.gru(my_input.float(), hidden_state)

        # print('a', my_input.shape)
        # U_t = matrix_mul(out_features, self.word_weight, self.word_bias)
        # U_w = matrix_mul(my_input, self.context_weight).permute(1, 0)
        # alpha_t = F.softmax(my_input)
        # weight_sum = element_wise_mul(out_features, output.permute(1, 0))
        print('out_features shape is: {}'.format(out_features.shape))
        print('h_n shape is: {}'.format(h_n.shape))
        U_t = torch.tanh_(self.attn(out_features))
        alpha_t = F.softmax(self.contx(U_t), dim=1)
        # check that it actually cycles multiplying the right weight with the right hidden features
        weighted_sum = (alpha_t * out_features).sum(1)
        print('finishing forward')
        return alpha_t.permute(0, 2, 1), weighted_sum

        # return U_t, alpha_t, weight_sum, out_features, h_n


lyrics = 'This is a test, please work\nPlease work I am begging you\nTonight is going to rain maybe'
emb_dim = 100
hidden_dim = 50
batch_size = 1
for verse in lyrics.split('\n'):
    emb_verse = embedder(verse, emb_dim)
    myGRU = WordAttNet(emb_dim, hidden_dim)
    # my_input=torch.randn(5, 2, 100)
    h0 = torch.randn(2, batch_size, hidden_dim)
    print(emb_verse.shape)
    #emb_verse = emb_verse.reshape(3, batch_size, 100)
    #output, out_features, hn = myGRU.forward(emb_verse, h0)
    alpha_t, weighted_sum = myGRU.forward(emb_verse, h0)

print('alpha_t shape is {}'.format(alpha_t.shape))
print('weighted_sum shape is {}'.format(weighted_sum.shape))
# print('hn shape is {}'.format(hn.shape))
# print('done')

######## IMPORTANT ########

# # input has size (seq_length x batch x input_size)
# my_input = torch.randn(5, 3, 60)
# # h0 has size (num_layers*num_directions x batch x hidden_size)
# h0 = torch.randn(2, 3, 50)
# # gru has size (input_size x hidden_size)
# gru = nn.GRU(60, 50, bidirectional=True)
# # output_f has shape (seq_length x batch x hidden_size * num_directions) 5, 3, 100
# # hn has shape (num_layers*num_directions x batch x hidden_size) 2, 3, 50
# output_f, hn = gru(my_input, h0)

################
