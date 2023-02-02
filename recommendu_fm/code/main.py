import time
import argparse
import numpy as np
import pandas as pd

from src import seed_everything
from src.data.context_data import Preprocessor, context_data_loader
# from src.data import context_data_load, context_data_split, context_data_loader
from src import FactorizationMachineModel, CatBoostModel
from sklearn.model_selection import StratifiedKFold

def main(args):
    seed_everything(args.SEED)

    ######################## DATA LOAD
    print(f'--------------- {args.MODEL} Load Data ---------------')
    preprocessor = Preprocessor(args)
    data = preprocessor.preprocess_train_dataset()

    ######################## TRAIN
    print(f'--------------- {args.MODEL} TRAINING ---------------')
    # model.train()
    skf = StratifiedKFold(n_splits = 5, shuffle = True)
    auc_scores = np.array([])
    for idx, (train_index, valid_index) in enumerate(skf.split(
                                        data['train'].drop(['label'], axis = 1),
                                        data['train']['label']
                                        )):
        #TODO: user별 validation 빼놓기
        #TODO: 
        data['X_train']= data['train'].drop(['label'], axis = 1).iloc[train_index]
        data['y_train'] = data['train']['label'].iloc[train_index]
        data['X_valid']= data['train'].drop(['label'], axis = 1).iloc[valid_index]
        data['y_valid'] = data['train']['label'].iloc[valid_index]
        data = context_data_loader(args, data)

        print(f'--------------- FOLD-{idx}, INIT {args.MODEL} ---------------')
        if args.MODEL=='FM':
            model = FactorizationMachineModel(args, data)
        elif args.MODEL == 'CatBoost':
            model = CatBoostModel(args, data)
        
        print(f'--------------- FOLD-{idx}, {args.MODEL} TRAINING ---------------')
        auc_score = model.train(fold_num = idx+1)
        auc_scores = np.append(auc_scores, auc_score)
        
        print(f"[FINAL AUC SCORES MEAN]: {auc_scores.mean()}")



if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--DATA_PATH', type=str, default='../data/', help='Data path를 설정할 수 있습니다.')
    arg('--SAVE_PATH', type = str, default = "/opt/ml/output/")
    arg('--MODEL', type=str, choices=['FM', 'CatBoost'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--DATA_SHUFFLE', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--TEST_SIZE', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--SEED', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    
    ############### TRAINING OPTION
    arg('--BATCH_SIZE', type=int, default=4, help='Batch size를 조정할 수 있습니다.')
    arg('--EPOCHS', type=int, default=30, help='Epoch 수를 조정할 수 있습니다.')
    arg('--LR', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--WEIGHT_DECAY', type=float, default=1e-3, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')
    arg('--PATIENCE', type = int, default = 3)

    arg('--ITERATIONS', type = int, default = 2000)
    arg('--DEPTH', type = int, default = 12)
    arg('--EVAL_METRIC', type = str, default = "AUC")
    ############### GPU
    arg('--DEVICE', type=str, default='cpu', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')

    ############### FM
    arg('--FM_EMBED_DIM', type=int, default=6, help='FM에서 embedding시킬 차원을 조정할 수 있습니다.')

    args = parser.parse_args()
    main(args)
