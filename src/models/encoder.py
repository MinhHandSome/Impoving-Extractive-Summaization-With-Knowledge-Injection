import math

import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from models.knowledge import cosine_similarity_matrix,lexrank_matrix,nmf
class Classifier(nn.Module):
    def __init__(self, hidden_size,hidden_dim):
        super(Classifier, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_dim,num_layers=12,batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        x , hidden = self.lstm(x).squeeze(-1)
        h = self.linear1(x)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)+2
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs,mask,similarity=None): #,similarity=None



        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask,similarity=similarity) #,similarity=similarity
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

# def nmf(X,  n_components=10, max_iter=100):
#     """Perform NMF on a non-negative matrix X."""
#     n, m = X.shape
#     W = np.random.rand(n, n_components)
#     H = np.random.rand(n_components, m)
#     # Normalize W to have maximum value of 1 along each column
#     W /= np.max(W, axis=0)
#     for i in range(max_iter):
#         # Update H
#         numerator = np.dot(W.T, X)
#         denominator = np.dot(np.dot(W.T, W), H)
#         H *= numerator / denominator
#         # Update W
#         numerator = np.dot(X, H.T)
#         denominator = np.dot(np.dot(W, H), H.T)
#         W *= numerator / denominator
#         # Normalize W to have maximum value of 1 along each column
#         W /= np.max(W, axis=0)
#     return W, H

class ExtTransformerEncoder(nn.Module):
    def __init__(self, args, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.args = args
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers

        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)



        self.wo = nn.Linear(d_model, 1, bias=True)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, top_vecs, mask,embeding_sent): #embeding_sent
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        embeding_sent = embeding_sent[:,:n_sents]
        if self.args.knowledge == 'cosine':
            matrix_knowledge = cosine_similarity_matrix(embeding_sent)
        elif self.args.knowledge == 'lexrank':
            matrix_knowledge = lexrank_matrix(embeding_sent)
        elif self.args.knowledge == 'nmf' :
            matrix_knowledge = []
            for matrix in cosine_similarity_matrix : 
                # transformed_matrix = (matrix + 1) / 2
                W, H = nmf(matrix, n_components=matrix.shape[1], max_iter=100)
                matrix_knowledge.append(W)
            matrix_knowledge = torch.tensor(matrix_knowledge)

        #Cosine similarity
        # embeding_sent = embeding_sent[:,:n_sents]
        # dot_product_matrix = torch.matmul(embeding_sent, embeding_sent.transpose(1, 2))
        # similarity_matrix = F.normalize(dot_product_matrix, dim=-1)
        # cosine_similarity_matrix = torch.matmul(similarity_matrix, similarity_matrix.transpose(1, 2))

        #NMF
        # matrix_factorization = []
        # for matrix in cosine_similarity_matrix : 
        #     transformed_matrix = (matrix + 1) / 2
        #     W, H = nmf(transformed_matrix, n_components=transformed_matrix.shape[1], max_iter=100)
        #     matrix_factorization.append(W)
        # matrix_factorization = torch.tensor(matrix_factorization)

        #Lexrank
        # matrix_lexrank = []
        # for matrix in embeding_sent:
        #     cosine_similarity_matrix = cosine_similarity(matrix)
            
        #     adjacency_matrix = np.where(cosine_similarity_matrix > 0.1, 1, 0)
        #     row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
        #     row_sums[row_sums==0] = 1
        #     connectivity_matrix = adjacency_matrix / row_sums
        #     n =adjacency_matrix.shape[0]  
        #     d = np.sum(adjacency_matrix,axis=0)
        #     d[d==0] = 1
        #     inv_d = 1.0 / d
        #     transition_matrix = adjacency_matrix * inv_d
            
        #     matrix_lexrank.append(connectivity_matrix)
        # matrix_lexrank = torch.tensor(matrix_lexrank)  
            
 
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb
        for i in range(self.num_inter_layers):
            if i < 1 : 
                x = self.transformer_inter[i](i, x, x, ~(mask),matrix_knowledge)  # all_sents * max_tokens * dim
            else :
                x = self.transformer_inter[i](i,x,x,~(mask))
        x = self.layer_norm(x)
        output = self.sigmoid(self.wo(x))
        # print(output.shape)
        sent_scores = output.squeeze(-1) * mask.float()
        return sent_scores

