import numpy as np
import pandas as pd
import argparse
import pickle
from collections import defaultdict
import time

from preprocess import RecommendTag

def parse_args():
    parser = argparse.ArgumentParser(

    )
    parser.add_argument("--data_dir", type=str, default = "/opt/ml/data/", help="crawled dataset")
    parser.add_argument("--topk", type=int, default = 4, help="recommend tag num")
    return parser.parse_args()   


def main(args):
    start1 = time.time()
    document = pd.read_csv(f"{args.data_dir}jk_documents_3_2.csv")
    jobkorea = pd.read_csv(f"{args.data_dir}jk_answers_without_samples_3_2.csv")

    with open(f"{args.data_dir}question_category_answerid.pkl","rb") as f:
        qcate_dict = pickle.load(f)

    matrix = np.load(f"{args.data_dir}answer_embedding_matrix.npy")
    
    sample_answer = 'LG Display에 지원하게 된 동기는 두 가지입니다.첫 번째, 디스플레이제조업계 최고의 위치를 차지하고 있음에도 항상 최선을 모습을 다하며 주위 이웃과 자연환경을 먼저 생각하는 LG Display의 이미지 때문입니다.\
    두 번째, LG Display는 진취적인 기상과 창의성을 가진 인재를 모집하는 오늘보다 내일이 좀 더 기대되는 곳이라고 생각됩니다.\
    그렇기에 LG Display의 매력에 흠뻑 빠져든 저 자신을 보았고 제가 가진 사회적 경험과 목표 달성을 위한 추진력 그리고 국가기술자격과 고교성적, \
    무엇보다 팀으로써 일할 수 있는 태도 등 제 모든 조건이 LG Display의 발전을 위해 존재해 왔다고 생각됩니다.이상 두 가지 이유로 지원하였습니다. 또한, 희망근무분야는 전공지식, 경험 등을 살려서 전기 보전 업무를 맡아 능력을 발휘하고 싶습니다.'

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
    rectag = RecommendTag(document, jobkorea, qcate_dict, matrix, example_user1["question_category"], example_user1["company"], 
                example_user1["favorite_company"], example_user1["job_large"], example_user1["answer"], args.topk)
    
    rectag.filtering()

    result = {"tag1" : rectag.gettag1(),"tag2" : rectag.gettag2(),"tag3" : rectag.gettag3(),"tag4" : rectag.gettag4(),"tag5" : rectag.gettag5()}
    print(result)
    print("recommend time : ", time.time()-start2)


if __name__ == '__main__':
    args = parse_args()
    main(args)