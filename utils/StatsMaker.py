from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
from utils.Timer import Timer
import torch
import numpy as np
import pandas as pd
import os

class StatisticsMaker:
    def __init__(self, model_name, gpu_id, main_gpu_id):
        self.main_gpu_id = main_gpu_id
        self.given_gpu_id = gpu_id
        # self.pbar = None
        self.base_dir = "./train_record/" + model_name + '/'
        self.params_dir = self.base_dir + "params/"
        self.csv_path = self.base_dir + "eval_scores.csv"
        self.best_model_path = self.base_dir + "best_model"
        self.epochs = []
        self.metrics_list = ["f1", "accuracy", "precision", "recall"]
        self.main_metric = "accuracy"
        self.scores_dict = defaultdict(dict)
        self.create_directory()
        self.epoch = -1
        self.epoch_losses = dict()
        self.loss = 0
        self.num_runs = 0
        self.timer = Timer(self.base_dir + "time_info.txt")
        #add results table

    def _active_method(func):
        def wrapper(self, *args, **kwargs):
            if self.main_gpu_id == self.given_gpu_id:
                return func(self, *args, **kwargs)
            else:
                # If GPU ID is not main, return None or handle it as needed
                return None
        return wrapper
         
    def _make_metrics_graph(self):
        plt.figure(figsize=(10, 6))
        for metric in self.metrics_list:
            plt.plot(*zip(*sorted(self.scores_dict[metric].items())), marker='o', label=metric, color=np.random.rand(3))
        plt.title('Evaluation while training')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics values')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.base_dir + "eval_scores.png")
    def _make_loss_graph(self):
        plt.figure(figsize=(10, 6))
        plt.plot(*zip(*sorted(self.epoch_losses.items())), marker='o', color=np.random.rand(3))
        plt.title('Losses while training')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        # plt.legend()
        plt.grid(True)
        plt.savefig(self.base_dir + "epoch_losses.png")

    def make_pbar(self, dataloader):
        if self.main_gpu_id == self.given_gpu_id:
            self.pbar = tqdm(dataloader, desc=f"Training Epoch {self.epoch}", position=0, leave=True)
            return self.pbar
        else:
            return dataloader

    @_active_method
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.epochs.append(epoch)
        self.num_runs = 0
        self.loss = 0

    @_active_method
    def add_loss(self, loss):
        self.loss += loss
        self.num_runs += 1

    @_active_method
    def set_description(self):
         self.pbar.set_description(f"Training Epoch {self.epoch}, Loss = {round(self.get_avg_loss(), 5)}")

    @_active_method
    def get_avg_loss(self):
        if self.num_runs != 0:
            return self.loss/self.num_runs
        else:
            return self.loss/1.0
    
    @_active_method
    def save_epoch_loss(self):
        if len(self.epoch_losses) == 0 and self.epoch != 0:
            df = pd.read_csv(self.base_dir + 'loss_info.csv')
            self.epoch_losses = df.to_dict(orient='dict')['0']
        self.epoch_losses[self.epoch] = self.get_avg_loss()
        self._make_loss_graph()
        df = pd.DataFrame.from_dict(self.epoch_losses, orient='index')
        df.to_csv(self.base_dir + 'loss_info.csv', index=False)


    @_active_method
    def epoch_time_measure(self):
        self.timer.time_measure(self.epoch)

    @_active_method
    def create_directory(self):
        directory1 = self.params_dir
        try:
            if not os.path.exists(directory1):
                os.makedirs(directory1)
        except OSError:
            print("Error: Failed to create the directory.")

    @_active_method
    def save_epoch_params(self, model):
        param_file = self.params_dir + "epoch_" + str(self.epoch) + ".pt"
        torch.save(model.state_dict(), param_file)

    @_active_method
    def save_last_params(self, model):
        param_file = self.params_dir + "last_epoch.pt"
        torch.save(model.state_dict(), param_file)
    
    @_active_method
    def process_metrics(self, model, scores):
        if len(self.scores_dict) == 0 and self.epoch != 0:
            df = pd.read_csv(self.csv_path)
            self.scores_dict = df.to_dict(orient='dict')
        for metric in self.metrics_list:
            score = scores[metric]
            self.scores_dict[metric][self.epoch] = score
        #save scores to csv file
        df = pd.DataFrame.from_dict(self.scores_dict, orient='index').T
        df.to_csv(self.csv_path, index=False)

        self._make_metrics_graph()
        cur_score = self.scores_dict[self.main_metric][self.epoch]
        best_model = True if cur_score == max(self.scores_dict[self.main_metric].values()) else False
        if best_model:
            torch.save(model.state_dict(), self.best_model_path)
            short_log = f"Best model achieved on epoch {self.epoch}, with {self.main_metric} score = {cur_score}\n"
            with open(self.base_dir + "short_log.txt", 'w') as file:
                file.write(short_log)