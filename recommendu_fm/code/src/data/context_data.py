import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset



def context_data_load(args):

    ######################## DATA LOAD
    df = pd.read_csv(args.DATA_PATH + 'train.csv')
    df = df.drop(columns=['answer_pro_good_cnt', 'answer_pro_bad_cnt', 'answer_view', 'rec_question_from_user'])
    test_df = pd.read_csv(args.DATA_PATH + 'test.csv')
    test_df = test_df.drop(columns=['answer_pro_good_cnt', 'answer_pro_bad_cnt', 'answer_view', 'rec_question_from_user', 'label'])
    answer_df = pd.read_csv(args.DATA_PATH + 'services_answer.csv')
    company_df = pd.read_csv(args.DATA_PATH + 'services_company.csv')
    joblarge_df = pd.read_csv(args.DATA_PATH + 'services_joblarge.csv')
    jobsmall_df = pd.read_csv(args.DATA_PATH + 'services_jobsmall.csv')
    majorlarge_df = pd.read_csv(args.DATA_PATH + 'services_majorlarge.csv')
    majorsmall_df = pd.read_csv(args.DATA_PATH + 'services_majorsmall.csv')
    questiontype_df = pd.read_csv(args.DATA_PATH + 'services_questiontype.csv')
    recommendtype_df = pd.read_csv(args.DATA_PATH + 'services_recommendtype.csv')

    # breakpoint()
    
    ids = df['user_id'].unique()
    answers = answer_df['answer_id'].unique()
    companies = company_df['company_id'].unique()
    job_larges = joblarge_df['job_large_id'].unique()
    job_smalls = jobsmall_df['job_small_id'].unique()
    major_larges = majorlarge_df['major_large_id'].unique()
    major_smalls = majorsmall_df['major_small_id'].unique()
    question_types = questiontype_df['question_type_id'].unique()
    recommend_types = recommendtype_df['rectype_id'].unique()
    career_types = ['신입', '경력']

    idx2user = {idx:id for idx, id in enumerate(ids)}
    user2idx = {id:idx for idx, id in idx2user.items()}
    
    idx2answer = {idx:id for idx, id in enumerate(answers)}
    answer2idx = {id:idx for idx, id in idx2answer.items()}
    
    idx2company = {idx:id for idx, id in enumerate(companies)}
    company2idx = {id:idx for idx, id in idx2company.items()}
    
    idx2joblarge = {idx:id for idx, id in enumerate(job_larges)}
    joblarge2idx = {id:idx for idx, id in idx2joblarge.items()}

    idx2jobsmall = {idx:id for idx, id in enumerate(job_smalls)}
    jobsmall2idx = {id:idx for idx, id in idx2jobsmall.items()}
    
    idx2majorlarge = {idx:id for idx, id in enumerate(major_larges)}
    majorlarge2idx = {id:idx for idx, id in idx2majorlarge.items()}

    idx2majorsmall = {idx:id for idx, id in enumerate(major_smalls)}
    majorsmall2idx = {id:idx for idx, id in idx2majorsmall.items()}

    idx2questiontype = {idx:id for idx, id in enumerate(question_types)}
    questiontype2idx = {id:idx for idx, id in idx2questiontype.items()}
    
    idx2recommendtype = {idx:id for idx, id in enumerate(recommend_types)}
    recommendtype2idx = {id:idx for idx, id in idx2recommendtype.items()}
    
    idx2careertype = {idx:id for idx, id in enumerate(career_types)}
    careertype2idx = {id:idx for idx, id in idx2careertype.items()}
    
    df['user_id'] = df['user_id'].map(user2idx)
    df['career_type'] = df['career_type'].map(careertype2idx)
    df['user_interesting_job_large'] = df['user_interesting_job_large'].map(joblarge2idx)
    df['user_major_small'] = df['user_major_small'].map(majorsmall2idx)
    df['answer_id'] = df['answer_id'].map(answer2idx)
    df['answer_major_small'] = df['answer_major_small'].map(majorsmall2idx)
    df['rectype_id'] = df['rectype_id'].map(recommendtype2idx)
    df['rec_company'] = df['rec_company'].map(company2idx)
    df['rec_job_large'] = df['rec_job_large'].map(joblarge2idx)
    df['rec_job_small'] = df['rec_job_small'].map(jobsmall2idx)        
    df['rec_question_type'] = df['rec_question_type'].map(questiontype2idx)
    test_df['user_id'] = test_df['user_id'].map(user2idx)
    test_df['career_type'] = test_df['career_type'].map(careertype2idx)
    test_df['user_interesting_job_large'] = test_df['user_interesting_job_large'].map(joblarge2idx)
    test_df['user_major_small'] = test_df['user_major_small'].map(majorsmall2idx)
    test_df['answer_id'] = test_df['answer_id'].map(answer2idx)
    test_df['answer_major_small'] = test_df['answer_major_small'].map(majorsmall2idx)
    test_df['rectype_id'] = test_df['rectype_id'].map(recommendtype2idx)
    test_df['rec_company'] = test_df['rec_company'].map(company2idx)
    test_df['rec_job_large'] = test_df['rec_job_large'].map(joblarge2idx)
    test_df['rec_job_small'] = test_df['rec_job_small'].map(jobsmall2idx)        
    test_df['rec_question_type'] = test_df['rec_question_type'].map(questiontype2idx)
    
    field_dims = np.array([len(idx2user), len(careertype2idx), len(joblarge2idx),
                          len(majorsmall2idx), len(answer2idx), len(majorsmall2idx),len(recommendtype2idx), 
                          len(company2idx), len(joblarge2idx), len(jobsmall2idx), len(questiontype2idx)], dtype=np.uint32)

    data = {
            'train':df,
            'test': test_df,
            'field_dims':field_dims
            }

    return data


def context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['label'], axis=1),
                                                        data['train']['label'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))


    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
