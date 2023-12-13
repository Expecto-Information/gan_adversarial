import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader

#custom
from GradAttacked import GradAttacked
from utils.StatsMaker import StatisticsMaker
from Evaluater import Evaluater


class GradTrainer:
    def __init__(
        self,
        grad_generator: torch.nn.Module,
        classifier: torch.nn.Module,
        model_name: str, 
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        gen_optimizer: torch.optim.Optimizer,
        gpu_id: int,
        accum_iter: int = 1,
        without_eval: bool = False
    ) -> None:
        self.gpu_id = gpu_id
        self.without_eval = without_eval
    
        self.stats = StatisticsMaker(model_name, gpu_id, gpu_id)
     
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.accum_iter = accum_iter
        self.count_iter = 0

        self.gen_optimizer = gen_optimizer

        self.eval_every = 1
        self.save_every = 1

        self.grad_generator = grad_generator.to(gpu_id)
        self.classifier = classifier.to(gpu_id)
        self.attacked_classifier = GradAttacked(self.grad_generator, self.classifier)
        
        self.evaluater = Evaluater(self.attacked_classifier, self.test_dataloader, self.gpu_id)

    def _run_batch(self, images, targets):
        images.requires_grad = True
        preds = self.classifier(images)
        self.classifier.zero_grad()
        cost = nnf.cross_entropy(preds, targets)
        cost.backward()

        target_directions = images.grad.sign()
        # target_directions = images.grad
        self.classifier.zero_grad()
        pred_directions = self.grad_generator(images).sign()

        # rounded_vector = torch.where(pred_directions >= 0.5, torch.tensor(1.0), torch.tensor(-1.0))
        # print("Squared Error: ", torch.pow(target_directions-rounded_vector, 2).sum())
        # print("Norm of target ", torch.pow(target_directions, 2).sum())
        # print("Norm of pred ", torch.pow(rounded_vector, 2).sum())

        loss = nnf.mse_loss(pred_directions, target_directions)
        # loss = nnf.binary_cross_entropy(pred_directions, target_directions)
        loss = loss / self.accum_iter
        self.count_iter += 1


        if not torch.isnan(loss):
            self.stats.add_loss(loss.item()*self.accum_iter)
            loss.backward()
        else:
            print("nanloss")

        if self.count_iter == self.accum_iter:
            self.gen_optimizer.step()
            self.gen_optimizer.zero_grad()
            self.count_iter = 0

    def _freeze_discriminator(self):
        for param in self.classifier.parameters():
            param.requires_grad = False

    def _run_train_gen_epoch(self):
        self.grad_generator.train() 
        self.classifier.train()
        for image, targets in self.stats.make_pbar(self.train_dataloader):
            image = image.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(image, targets)
            self.stats.set_description()
        self.stats.save_epoch_loss()

    def _run_eval(self):
        metrics = self.evaluater.eval()         
        self.stats.process_metrics(self.grad_generator, metrics)    

    def train(self, max_epochs: int, start_epoch = 0):
        # self._freeze_discriminator()

        for epoch in range(start_epoch, max_epochs):
            self.stats.set_epoch(epoch)
            self.stats.epoch_time_measure()
            self._run_train_gen_epoch()
            if epoch % self.save_every == 0:
                self.stats.save_last_params(self.grad_generator)
            if epoch % self.eval_every == 0 and not self.without_eval:
                self._run_eval()

            self.stats.epoch_time_measure()
            
    def evaluate(self):
        self.evaluater.eval()