import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np



class AttackedClassifier(nn.Module):
    def __init__(self, attacker, classifier):
        super(AttackedClassifier, self).__init__()

        self.attacker = attacker
        self.classifier = classifier

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

        noise_images = self.attacker(images)
        if draw_mode==True:
            self.draw_images(images.squeeze(0), noise_images.squeeze(0))

        preds = self.classifier(noise_images)

        return preds
