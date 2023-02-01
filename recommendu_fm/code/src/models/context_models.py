import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _FactorizationMachineModel

from src.utils import EarlyStopping

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

            bce_score = self.predict_train().item()
            early_stopping(bce_score, self.model)  

            print('fold_num: ', fold_num, 'epoch:', epoch, 'validation: bce:', bce_score)

            if early_stopping.early_stop:
                print("Early stopping")
                break




    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                
        return self.criterion(torch.Tensor(predicts), torch.Tensor(targets))


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts