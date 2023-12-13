import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
import os, random, argparse


# custom
from Classifier import ResNet18
from NoiseGen.Generator import *
from NoiseGen.GANTrainer import GANTrainer
from MakeDataset import MakeDataset

# reproducibility
def initialization(seed = 0):   
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # multi-GPU
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 


parser = argparse.ArgumentParser(description='Settings for PrefixModel training')
parser.add_argument('-n', '--exp_name', type=str, default='no_name', help='name of the experiment')
parser.add_argument('-device', '--device', type = int, default=0, help='device number to use (if available)')
parser.add_argument('-train_bs', '--train_bs', type = int, default=20, help='train batch size')
parser.add_argument('-test_bs', '--test_bs', type = int, default=20, help='test batch size')

args = parser.parse_args()



#------------------Settings--------------------
#reproducibility
random_seed=42
initialization(seed=random_seed)
print("random_seed :", random_seed)

total_epochs = 55
LR = 5e-3

TEST_BATCH_SIZE = 30
TRAIN_BATCH_SIZE = 30

base_dir = '/home/stud_valery/gan_adversarial/data'

train_dataset = MakeDataset(base_dir, 'training_set')
test_dataset = MakeDataset(base_dir, 'test_set')

train_dataloader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True, num_workers=8)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=TEST_BATCH_SIZE,
                              shuffle=False, num_workers=8)

#============Experiment================
torch.cuda.empty_cache()

MODEL_NAME = args.exp_name 

USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:'+ str(args.device) if USE_CUDA else 'cpu')
print("Using device: ", device)

#which generator to use
attacker = GeneratorVAE()

attacker_opt = AdamW(attacker.parameters(), lr=LR, weight_decay = 0.01)

classifier = ResNet18(num_classes=2)
params = torch.load('/home/stud_valery/gan_adversarial/train_record/no_normalization/best_model',\
                     map_location='cuda:'+str(args.device))
classifier.load_state_dict(params)

classifier_opt = AdamW(classifier.parameters(), lr=LR, weight_decay = 0.01)


trainer = GANTrainer(attacker, classifier, MODEL_NAME, train_dataloader, train_dataloader, attacker_opt, classifier_opt, args.device)
trainer.train(total_epochs)


torch.cuda.empty_cache()
#============Experiment================