import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as nnf
import torch


class FGSMClassifier(nn.Module):
    def __init__(self, classifier, device = 0, eps = 0.007):
        super(FGSMClassifier, self).__init__()

        self.device = device
        self.classifier = classifier
        self.eps = eps

    def process_tensor_img(self, img):
        numpy_img=img.detach().cpu().numpy()
         # Transpose the array to match the shape expected by matplotlib (H x W x C)
        img_array=np.transpose(numpy_img, (1, 2, 0))
        return img_array
    
    def draw_images(self, img1, img2):
        # Assuming img1 and img2 are your two images (numpy arrays)
        img1 = self.process_tensor_img(img1)
        img2 = self.process_tensor_img(img2)
        # Transpose the array to match the shape expected by matplotlib (H x W x C)

        # Create subplots
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.title('Original image')

        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.title('Noised image')

        plt.show()

    def forward(self, images, draw_mode=False):
        self.classifier.train()
        torch.set_grad_enabled(True)
        images.requires_grad = True
        preds = self.classifier(images)
        self.classifier.zero_grad()

        cost = nnf.cross_entropy(preds.to(self.device), preds.argmax(dim=1).to(self.device))
        cost.backward()

        noised_images = images + self.eps*images.grad.sign()

        if draw_mode==True:
            self.draw_images(images[0], noised_images[0])

        preds = self.classifier(noised_images)
        if self.training:
            return preds
        else:
            return preds.argmax(dim=1)
