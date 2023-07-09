import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn.functional as F
 
def cosine_similarity_matrix(embeding_sent):
    dot_product_matrix = torch.matmul(embeding_sent, embeding_sent.transpose(1, 2))
    similarity_matrix = F.normalize(dot_product_matrix, dim=-1)
    cosine_similarity_matrix = torch.matmul(similarity_matrix, similarity_matrix.transpose(1, 2))
    cosine_similarity_matrix = (cosine_similarity_matrix+1)/2
    return cosine_similarity_matrix



def lexrank_matrix(embeding_sent):
    matrix_lexrank = []
    for matrix in embeding_sent:
        cosine_similarity_matrix = cosine_similarity(matrix)
        
        adjacency_matrix = np.where(cosine_similarity_matrix > 0.1, 1, 0)
        row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums==0] = 1
        connectivity_matrix = adjacency_matrix / row_sums
        n =adjacency_matrix.shape[0]  
        d = np.sum(adjacency_matrix,axis=0)
        d[d==0] = 1
        inv_d = 1.0 / d
        transition_matrix = adjacency_matrix * inv_d
        
        matrix_lexrank.append(connectivity_matrix)
    matrix_lexrank = torch.tensor(matrix_lexrank)  
    return  matrix_lexrank


def nmf(X,  n_components=10, max_iter=100):
    """Perform NMF on a non-negative matrix X."""
    n, m = X.shape
    W = np.random.rand(n, n_components)
    H = np.random.rand(n_components, m)
    # Normalize W to have maximum value of 1 along each column
    W /= np.max(W, axis=0)
    for i in range(max_iter):
        # Update H
        numerator = np.dot(W.T, X)
        denominator = np.dot(np.dot(W.T, W), H)
        H *= numerator / denominator
        # Update W
        numerator = np.dot(X, H.T)
        denominator = np.dot(np.dot(W, H), H.T)
        W *= numerator / denominator
        # Normalize W to have maximum value of 1 along each column
        W /= np.max(W, axis=0)
    return W, H
