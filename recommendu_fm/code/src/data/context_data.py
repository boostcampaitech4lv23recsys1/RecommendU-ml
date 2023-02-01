import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder


class StandardScaler:
    def __init__(self):
        self.train_mean = None
        self.train_std = None

    def build(self, train_data):
        self.train_mean = train_data.mean()
        self.train_std = train_data.std()

    def normalize(self, df):
        return (df - self.train_mean) / self.train_std

def view2categ(x):
    x = int(x)
    if x < 12673:
        return 0
    elif x >= 12673 and x < 25064:
        return 1
    elif x >= 25064 and x < 79171:
        return 2
    else:
        return 3


def good2categ(x):
    x = int(x)
    if x < 1:
        return 0
    elif x >= 1 and x < 2:
        return 1
    elif x >= 2 and x < 3:
        return 2
    else:
        return 3

def bad2categ(x):
    x = int(x)
    if x < 1:
        return 0
    elif x >= 1 and x < 2:
        return 1
    elif x >= 2 and x < 3:
        return 2
    else:
        return 3

class Preprocessor():
    def __init__(self, args):
        self.args = args
        self.train_data_path = os.path.join(self.args.DATA_PATH, "fm_ver2.csv")
        
        self.answer2idx = LabelEncoder()
        self.rectype2idx = LabelEncoder()
        self.document2idx = LabelEncoder()

        # self.doc_rating_normalizer = StandardScaler()


    def _load_train_dataset(self, train_data_path = None):
        print("starting to load train data: ")
        train_data = pd.read_csv(self.train_data_path)
        train_data = train_data.loc[train_data['user'] != 1].reset_index(drop = True)
        
        print("train data shape: ")
        print(train_data.shape)

        answer_df = pd.read_csv(self.args.DATA_PATH + 'services_answer.tsv', sep = '\t')
        recommendtype_df = pd.read_csv(self.args.DATA_PATH + 'services_recommendtype.csv')
        document_df = pd.read_csv(self.args.DATA_PATH + 'services_document.csv')

        answers = answer_df['answer_id'].unique()
        print(answer_df['answer_id'])
        recommend_types = recommendtype_df['rectype_id'].unique()
        documents = document_df['document_id'].unique()
        
        self.answer2idx.fit(answers)
        self.rectype2idx.fit(recommend_types)
        self.document2idx.fit(documents)

        #원래대로라면 trainset과 valid를 나눈 후에 train에만 build하고 valid set에 normalize를 해야함.
        #아 rating -1때문에 쓰기가 어렵구나

        return train_data

    def preprocess_train_dataset(self):
        print("load train data to preprocess...")
        train_data = self._load_train_dataset()

        train_data['answer'] = self.answer2idx.transform(train_data['answer'])
        train_data['rectype'] = self.rectype2idx.transform(train_data['rectype'])
        train_data['document'] = self.document2idx.transform(train_data['document'])

        train_data['doc_view'] = train_data['doc_view'].map(view2categ)
        train_data['answer_pro_good_cnt'] = train_data['answer_pro_good_cnt'].map(good2categ)
        train_data['answer_pro_bad_cnt'] = train_data['answer_pro_bad_cnt'].map(good2categ)

        train_data = train_data[['answer', 'rectype', 'document', 'coin_company', \
                                'coin_jobsmall', 'coin_question_type', 'answer_pro_good_cnt',
                                'answer_pro_bad_cnt', 'doc_view', 'label'
                            ]]

        print(len(self.answer2idx.classes_))
        field_dims = np.array([len(self.answer2idx.classes_), len(self.rectype2idx.classes_),
                                    len(self.document2idx.classes_), 2, 2, 2, 4, 4, 4], dtype=np.uint32)
        
        data = {
            'train': train_data,
            'field_dims':field_dims
            }

        return data

        # return train_data, field_dims
    
    def preprocess_test_data(self, test_data):
        # windowing to get prev duration
        return dataset


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
    # test_dataset = TensorDataset(torch.LongTensor(data['test'].values))


    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader']= train_dataloader, valid_dataloader
    # data['test_dataloader'] = test_dataloader

    return data
