import copy
import numpy as np
import pandas as pd
import random

import torch

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics.pairwise import cosine_similarity
from similarity import content_based_filtering_cosine

class FeatureExtractor:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def match_question_top1(self, question_string, embedding_matrix) -> int:
        encoded_input = self.tokenizer(question_string, padding = True, truncation = True, return_tensors = 'pt')
        with torch.no_grad():
            output = self.model(**encoded_input)
            embedding = self.mean_pooling(output, encoded_input['attention_mask'])
        similarity = cosine_similarity(embedding, embedding_matrix)[0]
        top1_idx = np.argsort(similarity)[::-1][0].item()
        top1_sim = np.sort(similarity)[::-1][0].item()
        
        return top1_idx + 1, top1_sim


class Recommendation:
    def __init__(self, document, item, qcate_dict, matrix, embedder, question_category, company, 
                 job_large, job_small, answer, topk):
        
        #data
        self.document = document
        self.item = item
        self.qcate_dict = qcate_dict
        self.matrix = matrix

        # model
        self.embedder = embedder
        
        #user information
        self.question_category = str(question_category)
        self.company = company
        self.job_large = job_large
        self.job_small = job_small
        self.answer = None if len(answer) == 0 else answer
        
        #setting
        self.topk = topk
        
        #filtering
        self.fquestion = None
        self.fcompany = None
        self.fjob = None
    

    def filtering(self):
        self.fquestion = self.qcate_dict[self.question_category] #질문 필터링
        self.fcompany = list(self.document[self.document["company"] == self.company]["doc_id"])
        self.job_large = list(self.document[self.document["job_large"] == self.job_large]["doc_id"])
        self.job_small = list(self.document[self.document["job_small"] == self.job_small]["doc_id"])


    def _process_without_answer(self, tag):
        result = tag.sort_values(by=["weight_score", "pro_good_cnt", "doc_view"], ascending = [False, False, False])[["answer_id", "weight_score"]]
        answer_ids, scores = result['answer_id'], result['weight_score']
        answer_ids = list(answer_ids)[:10]
        scores = list(scores)[:10]
    
        high_score = scores[0]
        high_score_answers = [answer_id for answer_id, score in zip(answer_ids, scores) if score == high_score]
        residue = [answer_id for answer_id, score in zip(answer_ids, scores) if score != high_score]

        if len(high_score_answers) >= self.topk:
            return random.sample(high_score_answers, self.topk)
        else:
            result_ids = high_score_answers + random.sample(residue, self.topk - len(high_score_answers))
            return result_ids
    

    def recommend_with_company_jobtype(self):
        """
        Tag 1 : 질문 O / 회사 O / 직무 O
        """

        tag1 = self.item.copy()
        tag1['weight_score'] = np.zeros(len(self.item))
        tag1['weight_score'] += np.where(self.item['answer_id'].isin(self.fquestion), 2, 0)
        tag1['weight_score'] += np.where(self.item['doc_id'].isin(self.fcompany), 1, 0)
        tag1['weight_score'] += np.where(self.item['doc_id'].isin(self.job_large), 0.7, 0)
        tag1['weight_score'] += np.where(self.item['doc_id'].isin(self.job_small), 0.3, 0)
    
        if self.answer != None:
            tag1 = tag1.sort_values(by = 'weight_score', ascending = False).iloc[:10]
            tag1 = content_based_filtering_cosine(np.array(tag1["answer_id"]), self.matrix[tag1["answer_id"]], self.answer,
                                   self.embedder, self.topk)
            return list(tag1)
        else:
            return self._process_without_answer(tag1)
    

    def recommend_with_jobtype_without_company(self):
        """
        Tag 2 : 질문 O / 회사 x / 직무 O
        """
        
        tag2 = self.item.copy()
        tag2['weight_score'] = np.zeros(len(self.item))
        tag2['weight_score'] += np.where(self.item['answer_id'].isin(self.fquestion), 2, 0)
        tag2['weight_score'] += np.where(self.item['doc_id'].isin(self.job_large), 1, 0)
        tag2['weight_score'] += np.where(self.item['doc_id'].isin(self.job_small), 0.3, 0)
        
        if self.answer != None:
            tag2 = tag2.sort_values(by = 'weight_score', ascending = False).iloc[:10]
            tag2 = content_based_filtering_cosine(np.array(tag2["answer_id"]), self.matrix[tag2["answer_id"]], self.answer,
                                   self.embedder, self.topk)
            return list(tag2)

        else:
            return self._process_without_answer(tag2)
        
    
    
    def recommend_with_company_without_jobtype(self):
        """
        Tag 3 : 질문 O / 회사 O / 직무 X
        """
        tag3 = self.item.copy()
        tag3['weight_score'] = np.zeros(len(self.item))
        tag3['weight_score'] += np.where(self.item['answer_id'].isin(self.fquestion), 1, 0)
        tag3['weight_score'] += np.where(self.item['doc_id'].isin(self.fcompany), 2, 0)

        if self.answer != None:
            tag3 = tag3.sort_values(by = 'weight_score', ascending = False).iloc[:10]
            tag3 = content_based_filtering_cosine(np.array(tag3["answer_id"]), self.matrix[tag3["answer_id"]], self.answer,
                                   self.embedder, self.topk)
            return list(tag3)
        
        else:
            return self._process_without_answer(tag3)
    
    
    def recommed_based_popularity(self):
        """
        Tag 4 : popularity
        """
        tag4 = self.item[self.item["answer_id"].isin(self.fquestion)].sort_values('doc_view', ascending = False)
                             
        return list(tag4["answer_id"])[:self.topk]
    
    
    def recommend_based_expert(self):
        """
        Tag 5 : 전문가 평가
        """
        tag5_good = self.item[self.item["answer_id"].isin(self.fquestion)].sort_values(by=['pro_good_cnt',"pro_bad_cnt"], ascending = [False,True])
        tag5_bad = self.item[self.item["answer_id"].isin(self.fquestion)].sort_values(by=['pro_bad_cnt',"pro_good_cnt"], ascending = [False,True])
        
        tag5_good = list(tag5_good["answer_id"])[:self.topk//2]
        tag5_bad = list(tag5_bad["answer_id"])[:self.topk//2]
        
        return tag5_good, tag5_bad
    

    