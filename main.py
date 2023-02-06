import numpy as np
import pandas as pd
import pymysql
import ast
from tqdm import tqdm
from api import *
from fm_preprocess import *
import time


def main():
    start = time.time()
    conn = get_api()

    log_use_features = ['rec_timestamp', 'answer_id', 'rectype_id',
        'user_id', 'label', 'rec_log_id', 'question_from_user',
        'company_id', 'job_small_id', 'question_type_id']

    preprocess = Preprocess(log_use_features, conn)

    #### 답변 클릭 로그 기준으로 추천 버튼 클릭 로그 가져오기 ####
    fe_concat = preprocess.map_answerlog_last_reclog(get_answerlogs_api(conn), get_reclogs_api(conn))

    #### 노출된 데이터 항목을 활용하여 positive: 1, negative:0 label 생성 #### 
    preprocess.make_negative_answerlog(fe_concat)

    #### side information 병합 ####
    preprocess.merge_side_information_user()
    preprocess.merge_side_information_answer()
    preprocess.merge_side_information_job()
    preprocess._feature_engineering_data() #coin 계산시 변수 이름을 구분하기 위해 먼저 실행
    preprocess.make_coin_feature()

    ### get fe data ###
    fe_data = preprocess.get_fe_data()

    print(fe_data.head())
    print(fe_data.shape)
    print(fe_data.columns)
    print("total time : ", time.time() - start)


if __name__ == '__main__':
    main()