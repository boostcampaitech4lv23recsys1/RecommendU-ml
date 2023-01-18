import numpy as np
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline, Pipeline

from similarity import content_based_filtering_euclidean

class RecommendModel:
    sim_model_name = "jhgan/ko-sroberta-multitask"
    sim_tokenizer = AutoTokenizer.from_pretrained(sim_model_name)
    sim_model = AutoModel.from_pretrained(sim_model_name)


class RecommendTag(RecommendModel):
    def __init__(self, document, jobkorea, qcate_dict, matrix, question_category, company, 
                 favorite_company, job_large, answer, topk):
        #data
        self.document = document
        self.jobkorea = jobkorea
        self.qcate_dict = qcate_dict
        self.matrix = matrix
        
        #user information
        self.question_category = question_category
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
        
    #Tag 1 : 질문 O / 회사 O / 직무 O
    def gettag1(self):
        tag1 = self.jobkorea[(self.jobkorea["answer_id"].isin(self.fquestion)) 
                             & (self.jobkorea["doc_id"].isin(self.fcompany))
                             & (self.jobkorea["doc_id"].isin(self.job_large))]
        if self.answer:
            tag1 = content_based_filtering_euclidean(np.array(tag1["answer_id"]), self.matrix[tag1["answer_id"]], self.answer,
                                   super().sim_tokenizer, super().sim_model, self.topk-1)
            return list(tag1)
        else:
            temp = list(tag1.sort_values(by=["pro_good_cnt","doc_view"],ascending=[False,False])["answer_id"])[:30]
            if len(temp) >= self.topk:
                return random.sample(temp, self.topk)
            else:
                return temp
    
    #Tag 2 : 질문 O / 회사 x / 직무 O
    def gettag2(self):
        tag2 = self.jobkorea[(self.jobkorea["answer_id"].isin(self.fquestion)) 
                             & (self.jobkorea["doc_id"].isin(self.job_large))]
        
        if self.answer:
            tag2 = content_based_filtering_euclidean(np.array(tag2["answer_id"]), self.matrix[tag2["answer_id"]], self.answer,
                                   super().sim_tokenizer, super().sim_model, self.topk-1)
            return list(tag2)
        else:
            temp = list(tag2.sort_values(by=["pro_good_cnt","doc_view"],ascending=[False,False])["answer_id"])[:30]
            if len(temp) >= self.topk:
                return random.sample(temp, self.topk)
            else:
                return temp
        
    
    #Tag 3 : 질문 O / 회사 O / 직무 X
    def gettag3(self):
        tag3 = self.jobkorea[(self.jobkorea["answer_id"].isin(self.fquestion)) 
                             & (self.jobkorea["doc_id"].isin(self.fcompany))]
        
        if self.answer:
            tag3 = content_based_filtering_euclidean(np.array(tag3["answer_id"]), self.matrix[tag3["answer_id"]], self.answer,
                                   super().sim_tokenizer, super().sim_model, self.topk-1)
            return list(tag3)
        else:
            temp = list(tag3.sort_values(by=["pro_good_cnt","doc_view"],ascending=[False,False])["answer_id"])[:30]
            if len(temp) >= self.topk:
                return random.sample(temp, self.topk)
            else:
                return temp
    
    #Tag 4 : popularity
    def gettag4(self):
        tag4 = self.jobkorea[self.jobkorea["answer_id"].isin(self.fquestion)].sort_values('doc_view',ascending=False)
                             
        return list(tag4["answer_id"])[:self.topk]
    
    #Tag 5 : 전문가 평가
    def gettag5(self):
        tag5_good = self.jobkorea[self.jobkorea["answer_id"].isin(self.fquestion)].sort_values(by=['pro_good_cnt',"pro_bad_cnt"],ascending=[False,True])
        tag5_bad = self.jobkorea[self.jobkorea["answer_id"].isin(self.fquestion)].sort_values(by=['pro_bad_cnt',"pro_good_cnt"],ascending=[False,True])
        
        tag5_good = list(tag5_good["answer_id"])[:self.topk//2]
        tag5_bad = list(tag5_bad["answer_id"])[:self.topk//2]
        
        return tag5_good, tag5_bad
    

    