import torch
import numpy as np
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from XAI import IntegratedGradients, attribution_mask_processing, GradCAM
from models import VGGNB, NCNN


model = VGGNB()
#model = NCNN()
model.load_state_dict(torch.load('models\\best_VGGNB.pt', map_location=torch.device('cpu')))

size = 224
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size,size)),
            transforms.Normalize(
                mean=[0.36703529, 0.41083294, 0.50661294], std=[1, 1, 1])
        ])
path = 'Datasets\\Folds\\1\\Test\\ID343_UNIFESP_S32_pain.jpg'
target_class = 1
image = cv2.imread(path) # Load BGR image to VGGFace 
image_RGB = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (size,size))
#image = image_RGB # NCNN
image = image/255 # Normalize to [0-1]
image = np.float32(image)
image = transform(image)

baseline = torch.zeros((3,size,size))

ig = IntegratedGradients(model)
gradcam = GradCAM(model, model.VGGFace.features.conv5_3)
heatmap_ig = ig.attribution_mask(baseline, image, target_class=target_class, n_steps=50)
heatmap_gradcam = gradcam.attribution_mask(image.unsqueeze(dim=0), target_class)

res2_ig, alpha_channel_ig = attribution_mask_processing(heatmap_ig)
res2_cam, alpha_channel_cam = attribution_mask_processing(heatmap_gradcam)
import matplotlib
plt.subplot(2,2,1)
plt.imshow(heatmap_ig, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"]))
plt.imshow(image_RGB, alpha=0.5)

plt.subplot(2,2,2)
plt.imshow(image_RGB)
plt.imshow(res2_ig, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"]), alpha=alpha_channel_ig)

plt.subplot(2,2,3)
plt.imshow(heatmap_gradcam, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"]))
plt.imshow(image_RGB, alpha=0.5)

plt.subplot(2,2,4)
plt.imshow(image_RGB)
plt.imshow(res2_cam, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"]), alpha=alpha_channel_cam)

plt.show()