import torch
from tqdm import tqdm
import pandas as pd
from torchmetrics import F1Score, Accuracy, Precision, Recall

class Evaluater:
    def __init__(self, model, dataloader, device):
    
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.predicts = []
        self.ground_truth = []
        self.file_names = []

        self.f1 = F1Score(task="multiclass", num_classes=2, average="macro")
        self.acc = Accuracy(task="multiclass", num_classes=2)
        self.pr = Precision(task="multiclass", num_classes=2, average="macro")
        self.rc = Recall(task="multiclass", num_classes=2, average="macro")

            
    def _get_metrics(self):
        predicts = torch.cat(self.predicts, dim=0)
        ground_truth = torch.cat(self.ground_truth, dim=0)

        return {"f1": self.f1(predicts, ground_truth).item(),
                "accuracy": self.acc(predicts, ground_truth).item(),
                "precision": self.pr(predicts, ground_truth).item(),
                "recall": self.rc(predicts, ground_truth).item()}


    def _eval_pack(self, image, targets):
        with torch.no_grad():
            pred = self.model(image)

        self.predicts.append(pred.detach().cpu())
        self.ground_truth.append(targets)
    

    def _save_preds(self, dir_path):
      
        df = pd.DataFrame({'filename': self.file_names, 'preds': self.predicts})
        tsv_file_path = dir_path + '/preds.tsv'

        df.to_csv(tsv_file_path, sep='\t', index=False)

    def get_predicts(self, dir_path):
        self.model.eval().to(self.device)
        for image, file_names in tqdm(self.dataloader, desc="Calculating predictions..."):
            with torch.no_grad():
                pred = self.model(image.to(self.device))
                self.predicts.extend(pred.detach().tolist())
                self.file_names.extend(file_names)
        self._save_preds(dir_path)
        return self.predicts, self.file_names

    def eval(self):
        self.model.eval().to(self.device)

        for image, targets in tqdm(self.dataloader, desc="Evaluating..."):
            self._eval_pack(image.to(self.device), targets)

        return self._get_metrics()