import torch
from torch.nn import functional as nnf
from torch.utils.data import DataLoader

#custom
from NoiseGen.AttackedClassifier import AttackedClassifier
from utils.StatsMaker import StatisticsMaker
from utils.Evaluater import Evaluater

class GANTrainer:
    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        model_name: str, 
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        gen_optimizer: torch.optim.Optimizer,
        disc_optimizer: torch.optim.Optimizer,
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
        self.disc_optimizer = disc_optimizer

        self.eval_every = 1
        self.save_every = 1

        self.generator = generator.to(gpu_id)
        self.discriminator = discriminator.to(gpu_id)
        self.attacked_classifier = AttackedClassifier(self.generator, self.discriminator)
        
        self.evaluater = Evaluater(self.attacked_classifier, self.test_dataloader, self.gpu_id)

    def _run_gen_batch(self, images, targets):
        noise_images = self.generator(images)
        logits = self.discriminator(noise_images)

        flipped_targets = 1 - targets
        loss = nnf.cross_entropy(logits, flipped_targets)
        loss = loss / self.accum_iter
        self.count_iter += 1


        if not torch.isnan(loss):
            self.stats.add_loss(loss.item()*self.accum_iter)
            loss.backward()
        else:
            print("nanloss")

        if self.count_iter == self.accum_iter:
            # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
            self.gen_optimizer.step()
            self.gen_optimizer.zero_grad()
            # self.disc_optimizer.zero_grad()
            self.count_iter = 0

    def _freeze_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def _run_train_gen_epoch(self):
        self.generator.train() 
        self.discriminator.train()
        for image, targets in self.stats.make_pbar(self.train_dataloader):
            image = image.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_gen_batch(image, targets)
            self.stats.set_description()
        self.stats.save_epoch_loss()

    def _run_eval(self):
        metrics = self.evaluater.eval()         
        self.stats.process_metrics(self.generator, metrics)    

    def train(self, max_epochs: int, start_epoch = 0):
        self._freeze_discriminator()

        for epoch in range(start_epoch, max_epochs):
            self.stats.set_epoch(epoch)
            self.stats.epoch_time_measure()
            self._run_train_gen_epoch()
            if epoch % self.save_every == 0:
                self.stats.save_last_params(self.generator)
            if epoch % self.eval_every == 0 and not self.without_eval:
                self._run_eval()

            self.stats.epoch_time_measure()
            
    def evaluate(self):
        self.evaluater.eval()