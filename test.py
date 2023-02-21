import torch
import numpy as np
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from XAI import IntegratedGradients, attribution_mask_processing
from models import VGGNB


model = VGGNB()
model.load_state_dict(torch.load('models\\best_VGGNB.pt', map_location=torch.device('cpu')))

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.36703529, 0.41083294, 0.50661294], std=[1, 1, 1])
        ])
path = 'Datasets\\Folds\\2\\Test\\ID88_COPE_S08_pain.jpg'
target_class = 1
image = cv2.imread(path) # Load BGR image to VGGFace 
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image/255 # Normalize to [0-1]
image = np.float32(image)
image = transform(image)

baseline = torch.zeros((3,224,224))

ig = IntegratedGradients(model, 50)
heatmap = ig.attribution_mask(baseline, image, target_class=target_class)

res2, alpha_channel = attribution_mask_processing(heatmap)
import matplotlib
plt.imshow(image_RGB)
plt.imshow(np.squeeze(res2), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"]), alpha=alpha_channel)
plt.show()