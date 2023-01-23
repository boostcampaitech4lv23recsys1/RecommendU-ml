import numpy as np
import torch

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity


def content_based_filtering_euclidean(content_id_list, embedding_matrix, sentence, embedder, topn):
    encoded_input = embedder.tokenizer(sentence, padding = True, truncation = True, return_tensors = 'pt')
    
    with torch.no_grad():
        output = embedder.model(**encoded_input)
        embedding = embedder.mean_pooling(output, encoded_input['attention_mask'])
    
    sim_matrix = euclidean_distances(embedding, embedding_matrix)[0]
    sorted_idx = np.argsort(sim_matrix)[:topn]
    
    return content_id_list[sorted_idx]
    

def content_based_filtering_cosine(content_id_list, embedding_matrix, sentence, embedder, topn):
    encoded_input = embedder.tokenizer(sentence, padding = True, truncation = True, return_tensors = 'pt')
    with torch.no_grad():
        output = embedder.model(**encoded_input)
        embedding = embedder.mean_pooling(output, encoded_input['attention_mask'])
    
    sim_matrix = cosine_similarity(embedding, embedding_matrix)[0]
    sorted_idx = np.argsort(sim_matrix)[::-1][:topn]

    return content_id_list[sorted_idx]


def content_based_filtering_jaccard(content_id_list, token_set_dict, sentence, embedder, topn):
    encoded_input = embedder.tokenizer(sentence, padding = True, truncation = True, return_tensors = 'pt')
    sentence_tokens = set(encoded_input['input_ids'].squeeze().numpy())
    result = []
    for content_id in content_id_list:
        answer_tokens = token_set_dict[content_id]
        intersection = answer_tokens.intersection(sentence_tokens)
        similarity = len(intersection) / (len(sentence_tokens) + len(answer_tokens) - len(intersection))
        result.append(similarity)
    
    sorted_idx = np.argsort(np.array(result))[::-1][:topn]

    return content_id_list[sorted_idx]