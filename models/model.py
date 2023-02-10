import os
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

from ._models import _FactorizationMachineModel
from catboost import CatBoostClassifier
from utils import EarlyStopping

class FactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.criterion = nn.BCELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self, fold_num):
        # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        early_stopping = EarlyStopping(args=self.args, fold_num = fold_num, verbose=True)
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                self.model.zero_grad()
                fields, target = fields.to(self.device), target.to(self.device)

                y = self.model(fields)
                loss = self.criterion(y, target.float())
                # breakpoint()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            bce_score, pre_score, rec_score, acc_score, auc_score = self.predict_train(fold_num)
            early_stopping(auc_score, self.model)  

            print('fold_num: ', fold_num, 'epoch:', epoch, 'BCE:', bce_score, 'PRECISION:', pre_score, 'RECALL:', rec_score, 'ACC:',acc_score, 'AUC:', auc_score)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        return early_stopping.val_auc_max


    def plot_cfm_auc(self, cfm, fpr, tpr, auc_score, fold_num):
        LABELS = ['Interact', 'Non-interact']

        fig = plt.figure(figsize = (12,6))
        ax = fig.add_subplot(1,2,1)
        sns.heatmap(np.int16(cfm), xticklabels = LABELS, yticklabels = LABELS ,annot=True, cmap='Blues',fmt='g', annot_kws={'size' : 16})
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.title(f'Confusion Matrix')

        ax = fig.add_subplot(1,2,2)
        plt.plot(fpr, tpr, 
                lw=3, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='skyblue', lw=3,linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right", prop={'size' : 20})
        fig.tight_layout()
        plt.show()

        ppath = Path(os.path.join(self.args.SAVE_PATH, self.args.MODEL, f"fold{fold_num}", "result.png"))
        ppath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(ppath))

    def predict_train(self, fold_num):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        predicts_class = np.array(predicts)
        targets = np.array(targets)

        predicts_class = np.where(predicts_class >= 0.5, 1, 0)
        
        pre_score = precision_score(targets, predicts_class)
        rec_score = recall_score(targets, predicts_class)
        acc_score = accuracy_score(targets, predicts_class)
        cfm = confusion_matrix(targets, predicts_class, labels = [1, 0])

        fpr, tpr, thresholds = roc_curve(targets, np.array(predicts))
        auc_score = auc(fpr, tpr)
        self.plot_cfm_auc(cfm, fpr, tpr, auc_score, fold_num)

                
        return self.criterion(torch.Tensor(predicts), torch.Tensor(targets)).item(), pre_score, rec_score, acc_score, auc_score


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class CatBoostModel:

    def __init__(self, args, data):
        self.args = args
        self.train_data = data['train_dataloader']
        self.valid_data = data['valid_dataloader']
        self.cat_features = data['cat_features']

        params = {'iterations': args.ITERATIONS, 'learning_rate': 0.4, \
                    'depth': args.DEPTH, 'eval_metric': args.EVAL_METRIC, 'verbose': 100}
        self.model = CatBoostClassifier(**params)

    def train(self, fold_num):
        train_X, train_y = self.train_data
        valid_X, valid_y = self.valid_data

        print(f'CatBoost training... ', end='', flush=True)

        self.model.fit(train_X, train_y, eval_set = [(valid_X, valid_y)],\
            cat_features = self.cat_features, early_stopping_rounds = 100)#use_best_model = True,

        ppath = Path(os.path.join(self.args.SAVE_PATH, self.args.MODEL,f"fold{fold_num}",'checkpoint.cbm'))
        ppath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(ppath), format="cbm")

        """
        loaded_model = CatBoostClassifier()
        loaded_model.load_model("model.cbm", format="cbm")
        """
        
        print(f'done.')
        pre_score, rec_score, acc_score, auc_score = self.predict_train(fold_num)
        print('fold_num: ', fold_num, 'ACC:',acc_score, 'PRECISION:', pre_score, 'RECALL:', rec_score, 'AUC:', auc_score)

        return auc_score


    def plot_cfm_auc(self, cfm, fpr, tpr, auc_score, fold_num):
        LABELS = ['Interact', 'Non-interact']

        fig = plt.figure(figsize = (12,6))
        ax = fig.add_subplot(1,2,1)
        sns.heatmap(np.int16(cfm), xticklabels = LABELS, yticklabels = LABELS ,annot=True, cmap='Blues',fmt='g', annot_kws={'size' : 16})
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.title(f'Confusion Matrix')

        ax = fig.add_subplot(1,2,2)
        plt.plot(fpr, tpr, 
                lw=3, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='skyblue', lw=3,linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right", prop={'size' : 20})
        fig.tight_layout()
        plt.show()

        ppath = Path(os.path.join(self.args.SAVE_PATH, self.args.MODEL, f"fold{fold_num}", "result.png"))
        ppath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(ppath))


    def predict_train(self, fold_num):
        X, targets = self.valid_data

        predicts_class = self.model.predict(X)
        predicts = self.model.predict_proba(X)[:, 1]
        targets = np.array(targets)
        
        pre_score = precision_score(targets, predicts_class)
        rec_score = recall_score(targets, predicts_class)
        acc_score = accuracy_score(targets, predicts_class)
        cfm = confusion_matrix(targets, predicts_class, labels = [1, 0])

        fpr, tpr, thresholds = roc_curve(targets, np.array(predicts))
        auc_score = auc(fpr, tpr)
        self.plot_cfm_auc(cfm, fpr, tpr, auc_score, fold_num)

                
        return pre_score, rec_score, acc_score, auc_score


    def predict(self, dataloader):
        predicts = self.model.predict(dataloader[0])
        return predicts