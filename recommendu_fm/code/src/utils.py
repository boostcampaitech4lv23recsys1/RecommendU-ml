import os
import random
import numpy as np
import torch
import requests
from sklearn.model_selection import train_test_split
from pathlib import Path


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, args, fold_num, verbose=False, delta=0):
        self.args = args
        self.fold_num = fold_num
        self.patience = args.PATIENCE
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = -np.Inf
        self.delta = delta

    def __call__(self, val_auc, model):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        if self.verbose:
            print(
                f"Validation AUC increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ..."
            )
        ppath = Path(
            os.path.join(
                self.args.SAVE_PATH,
                self.args.MODEL,
                f"fold{self.fold_num}",
                'checkpoint.pt'
                    )
            )
        print(f"[earlystopping ppath]: {ppath}")

        ppath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(ppath))
        self.val_auc_max = val_auc
        
def send_model(from_path,to_path,model):
    files=open(from_path,'r',encoding='ISO-8859-1')
    obj={'model':model}
    upload = {'file':files}
    res = requests.post(to_path, files = upload,data=obj)