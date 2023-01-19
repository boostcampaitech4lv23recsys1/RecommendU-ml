import os
import json
import time
import argparse

import numpy as np
import pandas as pd

from preprocess import FeatureExtractor, Recommendation

def parse_args():
    parser = argparse.ArgumentParser(

    )
    parser.add_argument("--data_dir", type=str, default = "/opt/ml/data", help="crawled dataset")
    parser.add_argument("--topk", type=int, default = 4, help="recommend tag num")
    return parser.parse_args()   


def main(args):
    start1 = time.time()

    document = pd.read_csv(os.path.join(args.data_dir, "jk_documents_3_2.csv"), low_memory = False)
    item = pd.read_csv(os.path.join(args.data_dir, "jk_answers_without_samples_3_2.csv"), low_memory = False)

    embedder = FeatureExtractor(model_name = "jhgan/ko-sroberta-multitask")


    with open(f"{args.data_dir}question_cate_map_answerid.json","r") as f: #key: question_category, value(list): answer_id
        qcate_dict = json.load(f)

    answer_emb_matrix = np.load(f"{args.data_dir}answer_embedding_matrix.npy")
    
    f = open('sample_answer.txt', "r")
    sample_answer = f.read()

    # company o, answer o
    example_user1 = {"question_category" : 6, "company": "롯데IT테크(주)", "favorite_company":"네이버(주)", "job_large":"IT·인터넷", "answer":sample_answer}
    # company x, answer o
    example_user2 = {"question_category" : 6, "company": "", "favorite_company":"네이버(주)", "job_large":"IT·인터넷", "answer":sample_answer}
    # company o, answer x
    example_user3 = {"question_category" : 6, "company": "롯데IT테크(주)", "favorite_company":"네이버(주)", "job_large":"IT·인터넷", "answer":""}
    # company x, answer x
    example_user4 = {"question_category" : 6, "company": "", "favorite_company":"네이버(주)", "job_large":"IT·인터넷", "answer":""}

    
    print("data loader time : ", time.time()-start1)

    start2 = time.time()
    recommend = Recommendation(document, item, qcate_dict, answer_emb_matrix, embedder, 
                                example_user1["question_category"], example_user1["company"], example_user1["favorite_company"], example_user1["job_large"], example_user1["answer"], 
                                args.topk)
    
    recommend.filtering()

    result = {
            "tag1" : recommend.recommend_with_company_jobtype(),
            "tag2" : recommend.recommend_with_jobtype_without_company(),
            "tag3" : recommend.recommend_with_company_without_jobtype(),
            "tag4" : recommend.recommed_based_popularity(),
            "tag5" : recommend.recommend_based_expert()
            }

    print("=======태그별 추천 결과(answer_id)========")
    print(result)
    print("recommend time : ", time.time()-start2)


if __name__ == '__main__':
    args = parse_args()
    main(args)