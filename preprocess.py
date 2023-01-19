import numpy as np
import pandas as pd
import random

import torch

from transformers import AutoTokenizer, AutoModel

from similarity import content_based_filtering_cosine


class FeatureExtractor:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



class Recommendation:
    def __init__(self, document, item, qcate_dict, matrix, embedder, question_category, company, 
                 favorite_company, job_large, answer, topk):
        
        #data
        self.document = document
        self.item = item
        self.qcate_dict = qcate_dict
        self.matrix = matrix

        # model
        self.embedder = embedder
        
        #user information
        self.question_category = str(question_category)
        self.company = company if company else favorite_company
        self.job_large = job_large
        self.answer = answer if answer else None
        
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

    
    def recommend_with_company_jobtype(self):
        """
        Tag 1 : 질문 O / 회사 O / 직무 O
        """
        tag1 = self.item[(self.item["answer_id"].isin(self.fquestion)) 
                             & (self.item["doc_id"].isin(self.fcompany))
                             & (self.item["doc_id"].isin(self.job_large))]

        if self.answer:
            tag1 = content_based_filtering_cosine(np.array(tag1["answer_id"]), self.matrix[tag1["answer_id"]], self.answer,
                                   self.embedder, self.topk)
            return list(tag1)
        else:
            temp = list(tag1.sort_values(by=["pro_good_cnt","doc_view"],ascending=[False,False])["answer_id"])[:30]
            if len(temp) >= self.topk:
                return random.sample(temp, self.topk)
            else:
                return temp
    

    def recommend_with_jobtype_without_company(self):
        """
        Tag 2 : 질문 O / 회사 x / 직무 O
        """
        tag2 = self.item[(self.item["answer_id"].isin(self.fquestion)) 
                             & (self.item["doc_id"].isin(self.job_large))]
        
        if self.answer:
            tag2 = content_based_filtering_cosine(np.array(tag2["answer_id"]), self.matrix[tag2["answer_id"]], self.answer,
                                   self.embedder, self.topk)
            return list(tag2)
        else:
            temp = list(tag2.sort_values(by=["pro_good_cnt","doc_view"],ascending=[False,False])["answer_id"])[:30]
            if len(temp) >= self.topk:
                return random.sample(temp, self.topk)
            else:
                return temp
        
    
    
    def recommend_with_company_without_jobtype(self):
        """
        Tag 3 : 질문 O / 회사 O / 직무 X
        """
        tag3 = self.item[(self.item["answer_id"].isin(self.fquestion)) 
                             & (self.item["doc_id"].isin(self.fcompany))]
        
        if self.answer:
            tag3 = content_based_filtering_cosine(np.array(tag3["answer_id"]), self.matrix[tag3["answer_id"]], self.answer,
                                   self.embedder, self.topk)
            return list(tag3)
        else:
            result = list(tag3.sort_values(by=["pro_good_cnt","doc_view"],ascending=[False,False])["answer_id"])[:30]
            if len(result) >= self.topk:
                return random.sample(result, self.topk)
            else:
                return result
    
    
    def recommed_based_popularity(self):
        """
        Tag 4 : popularity
        """
        tag4 = self.item[self.item["answer_id"].isin(self.fquestion)].sort_values('doc_view',ascending=False)
                             
        return list(tag4["answer_id"])[:self.topk]
    
    
    def recommend_based_expert(self):
        """
        Tag 5 : 전문가 평가
        """
        tag5_good = self.item[self.item["answer_id"].isin(self.fquestion)].sort_values(by=['pro_good_cnt',"pro_bad_cnt"],ascending=[False,True])
        tag5_bad = self.item[self.item["answer_id"].isin(self.fquestion)].sort_values(by=['pro_bad_cnt',"pro_good_cnt"],ascending=[False,True])
        
        tag5_good = list(tag5_good["answer_id"])[:self.topk//2]
        tag5_bad = list(tag5_bad["answer_id"])[:self.topk//2]
        
        return tag5_good, tag5_bad
    

    