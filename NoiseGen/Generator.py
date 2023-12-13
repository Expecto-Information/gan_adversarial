

import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, max_pool):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if max_pool==1:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class GeneratorCNN(nn.Module):
    def __init__(self):
        super(GeneratorCNN, self).__init__()
        # Encoder
        self.conv_blocks = nn.ModuleList()

        #in_channels, out_channels, kernel_size, stride, max_pool(1 or 0)
        conv_params = [[3,32,3,1,0], [32,32,3,1,0], [32,64,3,1,0], [64,64,3,1,0],
                       [64,128,3,1,0], [128,64,3,1,0], [64,32,3,1,0], [32,3,3,1,0]]

        for param in conv_params:
            conv_block = ConvBlock(*param)
            self.conv_blocks.append(conv_block)
        
    def forward(self, tensor_images):
        # Encoder
        
        x = tensor_images.clone()

        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # print(x.shape)
        noise = x

        # current_sum = noise.sum(dim=(1, 2, 3))
        # norm_2 = torch.sum(torch.square(noise), dim=(1, 2, 3))
        
        normalized_noise = torch.zeros_like(noise)
        for i in range(len(noise)):
            normalized_noise[i] = noise[i]/torch.sqrt(torch.sum(torch.square(noise[i])))*50

        # norm_2 = torch.sum(torch.square(normalized_noise), dim=(1, 2, 3))
        # print(norm_2)

        # normalized_noise = self.normalize(noise)
        
        if self.training:
            noised_image = tensor_images + noise
        else:
            noised_image = tensor_images + normalized_noise

        return noised_image



class GeneratorVAE(nn.Module):
    def __init__(self):
        super(GeneratorVAE, self).__init__()

        self.latent_features = 512

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_features=14*14*256, out_features=1024),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features = self.latent_features, out_features = 1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features = 256*14*14),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 14, 14)),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )
        
        self.fc_mu = nn.Linear(1024, self.latent_features)
        self.fc_logvar = nn.Linear(1024, self.latent_features)
      
        self.target_sum = 100.0

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, tensor_images):
        
        pre_latent = self.encoder(tensor_images)

        mu = self.fc_mu(pre_latent)
        # mu = mu/mu.sum()
        log_var = self.fc_logvar(pre_latent)
        # log_var = log_var/log_var.sum()


        # mu = mu/mu.sum()
        # log_var = log_var/log_var.sum()

        sampled_vec = self.reparameterize(mu, log_var)
        
        noise = self.decoder(sampled_vec)
        

        # # current_sum = noise.sum(dim=(1, 2, 3))
        # # norm_2 = torch.sum(torch.square(noise), dim=(1, 2, 3))
        
        normalized_noise = torch.zeros_like(noise)
        for i in range(len(noise)):
            normalized_noise[i] = noise[i]/torch.sqrt(torch.sum(torch.square(noise[i])))*15

        # noised_image = tensor_images + noise

        return normalized_noise

