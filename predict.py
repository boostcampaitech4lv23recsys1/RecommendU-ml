import os
import json
import time
import argparse

import numpy as np
import pandas as pd

from preprocess import FeatureExtractor, Recommendation


MODEL_NAME = "jhgan/ko-sroberta-multitask"


def parse_args():
    parser = argparse.ArgumentParser(

    )
    parser.add_argument("--data_dir", type=str, default = "/opt/ml/data", help="crawled dataset")
    parser.add_argument("--topk", type=int, default = 4, help="recommend tag num")
    return parser.parse_args()   


def main(args):
    # Example
    input_question = "입사 후 포부 : 입사 후 10년 동안의 회사생활 시나리오와 그것을 추구하는 이유를 기술해주세요."
    start1 = time.time()
    sim = None

    document = pd.read_csv(os.path.join(args.data_dir, "jk_documents_3_2.csv"), low_memory = False)
    item = pd.read_csv(os.path.join(args.data_dir, "jk_answers_without_samples_3_2.csv"), low_memory = False)
    answer_emb_matrix = np.load(os.path.join(args.data_dir, "answer_embedding_matrix.npy"))
    question_emb_matrix = np.load(os.path.join(args.data_dir, "question_embedding_matrix.npy"))

    """
    DEBUG
    """
    with open(os.path.join(args.data_dir, "num2question.json"), 'r') as f: #key: question_category, value(list): answer_id
        num2question = json.load(f)

    embedder = FeatureExtractor(model_name = MODEL_NAME)
    if isinstance(question_category, str):
        question_category, sim = embedder.match_question_top1(input_question, question_emb_matrix)
    
    """
    if sim is not None and sim ~~ < XX:
        return
    """

    with open(os.path.join(args.data_dir, "question_cate_map_answerid.json"), 'r') as f: #key: question_category, value(list): answer_id
        qcate_dict = json.load(f)
        
    example_user = {"question_category" : 5, "company": "(주)LG화학",\
         "favorite_company":"네이버(주)", "job_large": "연구개발·설계", "job_small":"반도체·디스플레이", "answer":"'공정개선경험과 전공지식'저는 공정엔지니어에게 필요한 것은 화학공정에 대한 지식과 그것을 바탕으로 생산량과 에너지 효율을 향상시킬 수 있는 능력이라고 생각합니다. 저는 화학공장 설계프로젝트에서 공정개선으로 생산량을 20%향상시킨 경험이 있습니다. 처음에는 원하는 만큼 생산량이 안 나왔지만 DMAIC기법을 사용하여 공정데이터를 분석하여고 메탄올이 낭비되고 있다는 것을 파악하였습니다. "}

    print("data loader time : ", time.time() - start1)
    
    question_category, company, favorite_company, job_large, job_small, answer = example_user.values()

    start2 = time.time()
    recommend = Recommendation(document, item, qcate_dict, answer_emb_matrix, embedder, 
                                question_category, company, favorite_company, job_large, job_small, answer, 
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
    print("recommend time : ", time.time()-start2, '\n')
        


if __name__ == '__main__':
    args = parse_args()
    main(args)