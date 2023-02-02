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

    #### 답변 클릭 로그 기준으로 추천 버튼 클릭 로그 가져오기 ####
    fe_concat = map_answerlog_last_reclog(get_answerlogs_api(conn), get_reclogs_api(conn))

    #### 노출된 데이터 항목을 활용하여 positive: 1, negative:0 label 생성 #### 
    fe_data = make_negative_answerlog(fe_concat)

    #### side information 병합 ####
    fe_data = merge_side_information(conn, fe_data)

    print(fe_data.head())
    print(fe_data.shape)
    print(fe_data.columns)
    print("total time : ", time.time() - start)


if __name__ == '__main__':
    main()