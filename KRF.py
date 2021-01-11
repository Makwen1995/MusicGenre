

import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuralNetwork import NeuralNetwork
from GCN import RelationNet

class Attention(nn.Module):

    def __init__(self, in_features, hidden_size=None, max_sents=None):
        super(Attention, self).__init__()
        if hidden_size is None:
            hidden_size = in_features

        self.max_sents = max_sents
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, X_text):
        act = F.tanh(self.linear1(X_text))
        affine2 = self.linear2(act)
        score = F.softmax(affine2, dim=1)
        score = self.dropout(score)
        attention = torch.sum(score*X_text, dim=1)
        return attention


class KRF(NeuralNetwork):

    def __init__(self, config):
        super(KRF, self).__init__()
        self.patience = 0
        self.config = config
        self.max_comments = config['max_comments']
        self.max_words = config['max_len']
        self.bsz = config['batch_size']
        embedding_weights = torch.FloatTensor(config['embedding_weights'])
        V, D = embedding_weights.size()

        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=embedding_weights) #
        self.label_embedding = nn.Embedding(config["C"], 200)

        self.word_bigru = nn.GRU(input_size=D, hidden_size=100, batch_first=True, bidirectional=True)
        self.sent_bigru = nn.GRU(input_size=200, hidden_size=100, batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config['dropout'])
        self.word_attention = Attention(200, 100)
        self.sent_attention = Attention(200, 100)

        self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, config["C"])
        self.gcn = RelationNet(num_classes=config["C"], task=config['task'], adj_file=config['adj_file'])
        print(self)


    def word_level_rnn(self, X_word):
        '''
        :param X_word: (batch_size*max_sents, max_words, d)
        :return:
        '''
        X_sent, _ = self.word_bigru(X_word)
        X_sent = self.word_attention(X_sent)
        return X_sent


    def sentence_level_rnn(self, X_comment):
        '''
        :param X_sent: (batch_size, max_sents, d)
        :return:
        '''
        X_comment, _ = self.sent_bigru(X_comment)
        X_comment = self.sent_attention(X_comment)
        return X_comment


    def forward(self, X_text_idx):
        '''
        :param X_text size: (batch_size, max_comments, max_words)
        :return:
        '''
        # then lookup embedding  (batch_size, max_sents, max_words, D)
        X_word = self.word_embedding(X_text_idx)

        # reshape X_text to (batch_size*max_sents, max_words, D)
        X_word = X_word.view([-1, self.max_words, X_word.size(-1)])
        X_comment = self.word_level_rnn(X_word)

        # (batch_size, max_sents, nb_filters* len(Ks))
        X_comment = X_comment.view([-1, self.max_comments, X_comment.size(-1)])
        # (batch_size, nb_filters* len(Ks))
        X_text = self.sentence_level_rnn(X_comment)

        d1 = self.relu(self.fc1(X_text))

        Xt_logit = self.gcn(d1, self.label_embedding.weight)
        # Xt_logit = self.fc2(d1)
        return Xt_logit.sigmoid()





