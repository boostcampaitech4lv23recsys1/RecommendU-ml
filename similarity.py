import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline, Pipeline
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def content_based_filtering_euclidean(content_id_list, embedding_matrix, sentence, tokenizer, model, topn):
    topn=11 if topn is None else topn+1
    encoded_input = tokenizer(sentence, padding = True, truncation = True, return_tensors = 'pt')
    
    with torch.no_grad():
        output = model(**encoded_input)
        embedding = mean_pooling(output, encoded_input['attention_mask'])
    
    sim_matrix = euclidean_distances(embedding, embedding_matrix)[0]
    sorted_idx = np.argsort(sim_matrix)[:topn]
    
    return content_id_list[sorted_idx]
    

def content_based_filtering_cosine(content_id_list, embedding_matrix, sentence, tokenizer, model, topn):
    topn=11 if topn is None else topn+1
    encoded_input = tokenizer(sentence, padding = True, truncation = True, return_tensors = 'pt')
    with torch.no_grad():
        output = model(**encoded_input)
        embedding = mean_pooling(output, encoded_input['attention_mask'])
    sim_matrix = cosine_similarity(embedding, embedding_matrix)[0]
    print(max(sim_matrix), min(sim_matrix), sim_matrix.shape)
    sorted_idx = np.argsort(sim_matrix)[::-1][:topn]
    print(f"similarity check: {sim_matrix[sorted_idx]}")
    return content_id_list[sorted_idx]


def content_based_filtering_jaccard(content_id_list, token_set_dict, sentence, tokenizer, topn):
    topn=11 if topn is None else topn+1
    encoded_input = tokenizer(sentence, padding = True, truncation = True, return_tensors = 'pt')
    sentence_tokens = set(encoded_input['input_ids'].squeeze().numpy())
    result = []
    for content_id in content_id_list:
        answer_tokens = token_set_dict[content_id]
        intersection = answer_tokens.intersection(sentence_tokens)
        similarity = len(intersection) / (len(sentence_tokens) + len(answer_tokens) - len(intersection))
        result.append(similarity)
    sorted_idx = np.argsort(np.array(result))[::-1][:topn]
    print(f"similarity check: {np.array(result)[sorted_idx]}")
    return content_id_list[sorted_idx]