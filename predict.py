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
    question_category = "입사 후 포부 : 입사 후 10년 동안의 회사생활 시나리오와 그것을 추구하는 이유를 기술해주세요."
    start1 = time.time()

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
        question_category = embedder.match_question_top1(question_category, question_emb_matrix)
        

    with open(os.path.join(args.data_dir, "question_cate_map_answerid.json"), 'r') as f: #key: question_category, value(list): answer_id
        qcate_dict = json.load(f)
        

    
    f = open('sample_answer.txt', "r")
    sample_answer = f.read()

    # company o, answer o
    # example_user1 = {"question_category" : 6, "company": "교보생명", "favorite_company":"네이버(주)", "job_small":"응용프로그래머", "answer":sample_answer}
    # # company x, answer o
    # example_user2 = {"question_category" : 6, "company": "", "favorite_company":"네이버(주)", "job_small":"응용프로그래머", "answer":sample_answer}
    # # company o, answer x
    # example_user3 = {"question_category" : 6, "company": "롯데IT테크(주)", "favorite_company":"네이버(주)", "job_small":"응용프로그래머", "answer":""}
    # example_user4 = {"question_category" : 6, "company": "㈜KB데이타시스템", "favorite_company":"네이버(주)", "job_small":"경리·회계·결산", "answer":""}
    # example_user5 = {"question_category" : 10, "company": "(주)카카오", "favorite_company":"네이버(주)", "job_small":"경리·회계·결산", "answer":""}
    # example_user6 = {"question_category" : 8, "company": "(주)카카오", "favorite_company":"네이버(주)", "job_small":"경리·회계·결산", "answer":""}
    # # (학)연세대학교연세의료원
    # example_user = {"question_category" : 18, "company": "SK하이닉스(주)", "favorite_company":"네이버(주)", "job_large":"IT·인터넷", 
    # "answer":"최근 **전자 설명회를 통해 만나 뵈었던 VC사업본부 담당 리크루터께서는 **전자 VC사업본부는 최근 신설되었지만 기존부터 관련 연구가 진행되어 왔으며 회사가 가지고 있는 지능형 자동차 부품에 대한 기술경쟁력은 세계적 수준이라 하셨고 VC사업본부의 경쟁사는 국내의 자동차 부품업체가 아닌 보쉬와 컨티넨탈과 같은 세계적인 부품업체라는 점과 **그룹 계열사들이 가지고 있는 다양한 기술력을 통해 계열사의 협업을 바탕으로 성장 가능성 또한 무궁무진 하다는 이야기를 하셨습니다."}
    example_user = {"question_category" : 6, "company": "(주)SPC클라우드",\
         "favorite_company":"네이버(주)", "job_large": "IT·인터넷", "job_small":"시스템프로그래머", "answer":""}

    print("data loader time : ", time.time()-start1)
    
    print("[DEBUG]")
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
    print(example_user)
    print(result)
    print("recommend time : ", time.time()-start2, '\n')
    
    print(f"[QUESTION CATEGORY]: {num2question[str(question_category)]}")
    for idx, recommend_tag in enumerate(result.keys()):
        print(f"[TAG {idx + 1}]", '\n')
        if idx == 3:
            # TODO: list 2개 input
            break
        else:
            for answer_id in result[recommend_tag]:
                print(f"[QUESTION]: {item.iloc[answer_id]['question']}\n")
                print(f"[ANSWER]: {item.iloc[answer_id]['answer']}\n")
            print("-" * 250)
            print('\n\n')
        


if __name__ == '__main__':
    args = parse_args()
    main(args)