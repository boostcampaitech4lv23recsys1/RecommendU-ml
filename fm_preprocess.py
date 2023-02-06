import numpy as np
import pandas as pd
import pymysql
import ast
from tqdm import tqdm
import time
import api

class Preprocess():
    def __init__(self, log_use_features):
        self.log_use_features = log_use_features
        self.fe_data = pd.DataFrame()


    def map_answerlog_last_reclog(self, answerlogs, reclogs):
        answerlogs["label"] = 1
        answerlogs = answerlogs.rename(columns={'timestamp':'answer_timestamp'})

        _last_reclog = pd.DataFrame(columns=reclogs.columns)

        for row, value in answerlogs.iterrows():
            temp = reclogs[(reclogs["user"]==value["user"]) & (reclogs["timestamp"]<value["answer_timestamp"])].iloc[-1]
            _last_reclog = _last_reclog.append(temp, ignore_index=True)
        
        _last_reclog.rename(columns={'user':'rec_user_id'},inplace=True)


        fe_concat = pd.concat([answerlogs, _last_reclog], axis=1)
        fe_concat= fe_concat.drop(columns=['answer_timestamp'])
        fe_concat.rename(columns={'timestamp':'rec_timestamp'},inplace=True)

        return fe_concat


    def make_negative_answerlog(self, fe_concat):
        fe_data = fe_concat[self.log_use_features]

        rec_answer_id = fe_data.groupby('rec_log_id')['answer'].apply(list)

        last_rec_log = 0
        for row, value in tqdm(fe_concat.iterrows()):
            watched = rec_answer_id[value["rec_log_id"]]
            
            if value["rec_log_id"] == last_rec_log: #같은 추천버튼 로그로부터 오면 pass
                continue

            if value["impressions"]: #impressions이 존재할 때
                impressions = ast.literal_eval(value["impressions"])
                
                for i in range(20):
                    if impressions[i] in watched: #노출된 answer_id 중 클릭된 answer_id는 pass
                        continue
                        
                    if i in [18,19]:
                        rectype = '7000006'
                    else:
                        rectype = '700000' + str((i//4) +1)
                    
                    temp = [value["rec_timestamp"],impressions[i], rectype, value["user"],0, value["rec_log_id"], value["question_from_user"],value["company"], value["job_small"], value["question_type"]]
                    
                    fe_data = fe_data.append(pd.Series(temp, index=fe_data.columns),ignore_index=True)            
            last_rec_log = value["rec_log_id"]
        
        fe_data.rename(columns={"rec_timestamp":"timestamp", "answer":"answer","rectype":"rectype","user":"user",
                "company":"rec_company", "job_small":"rec_job_small","question_type":"rec_question_type"}, inplace=True)

        self.fe_data = fe_data

    def merge_side_information_user(self):
        #### user data merge ####
        user_data = api.load_user()

        use_user_data = user_data[["id", "career_type", "interesting_job_large","major_small"]]
        use_user_data.rename(columns = {'major_small' : 'user_major_small', "id":"user","interesting_job_large":"user_job_large", "career_type":"user_career_type"}, inplace = True)

        self.fe_data = self.fe_data.merge(use_user_data,how="left", on="user")


    def merge_side_information_answer(self):
        #### answer merge ###
        answers = api.load_answer()

        documents = api.load_document()
        answer_with_doc = answers.merge(documents[['document_id', 'company', 'job_small', 'major_small', "pro_rating"]], on='document_id')
        
        answer_with_doc.columns = ['answer', 'user_good_cnt', 'user_bad_cnt', 'answer_pro_good_cnt', 'answer_pro_bad_cnt',
                'doc_view', 'user_view', 'document', 'user_impression_cnt', 'answer_question_types','doc_company_id', 'doc_job_small_id', 'doc_major_small_id', "doc_pro_rating"]
        
        self.fe_data = self.fe_data.merge(answer_with_doc,how="left", on="answer")

        self.fe_data.rename(columns={"doc_major_small_id":"doc_major_small", "doc_major_large_id":"doc_major_large", 
                "doc_company_id":"doc_company", "doc_job_small_id":"doc_job_small"}, inplace=True)


    def merge_side_information_job(self):        
        #### job large merge ###
        jobsmall = api.load_job()
        rec_jobsmall = jobsmall[["job_small_id", "job_large"]]
        rec_jobsmall.rename(columns={"job_small_id":"rec_job_small", "job_large":"rec_job_large"}, inplace=True)

        doc_jobsmall = jobsmall[["job_small_id", "job_large"]]
        doc_jobsmall.rename(columns={"job_small_id":"doc_job_small", "job_large":"doc_job_large"}, inplace=True)

        self.fe_data = self.fe_data.merge(rec_jobsmall, how="left", on="rec_job_small")
        self.fe_data = self.fe_data.merge(doc_jobsmall, how="left", on="doc_job_small")

    
    def _feature_engineering_data(self):
        #### 정렬 및 rename ###
        self.fe_data = self.fe_data.sort_values(by=['rec_log_id'])

        self.fe_data = self.fe_data[["user","user_career_type","user_job_large","user_major_small","rectype","rec_company","rec_job_large","rec_job_small","rec_question_type",
            "answer","answer_pro_good_cnt","answer_pro_bad_cnt","answer_question_types","document","doc_view","doc_company","doc_job_large","doc_job_small","doc_major_small", "doc_pro_rating","label"]]


    def make_coin_feature(self):
        #### coin 추가 ####
        self.fe_data.loc[self.fe_data["rec_company"]==self.fe_data["doc_company"], "coin_company"] = 1
        self.fe_data.loc[self.fe_data["rec_company"]!=self.fe_data["doc_company"], "coin_company"] = 0

        self.fe_data.loc[self.fe_data["rec_job_large"]==self.fe_data["doc_job_large"], "coin_joblarge"] = 1
        self.fe_data.loc[self.fe_data["rec_job_large"]!=self.fe_data["doc_job_large"], "coin_joblarge"] = 0

        self.fe_data.loc[self.fe_data["rec_job_small"]==self.fe_data["doc_job_small"], "coin_jobsmall"] = 1
        self.fe_data.loc[self.fe_data["rec_job_small"]!=self.fe_data["doc_job_small"], "coin_jobsmall"] = 0

        ### 사용자가 선택한 질문이 answer question type에 포함되는지 ####
        for i in range(len(self.fe_data)):
            answer = self.fe_data.loc[i, "answer_question_types"].split(",")
            if str(self.fe_data.loc[i, 'rec_question_type']) in answer:
                self.fe_data.loc[i, 'coin_question_type'] = 1
            else:
                self.fe_data.loc[i, 'coin_question_type'] = 0


    def get_fe_data(self):
        return self.fe_data