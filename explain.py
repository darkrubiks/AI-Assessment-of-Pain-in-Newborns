"""
explain.py

Author: Leonardo Antunes Ferreira
Date: 16/03/2023

This is just an example file on how to run the XAI methods on any model. For this
example the VGGNB model is explained using both GradCAM and Integrated Gradients.
"""
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from XAI import IntegratedGradients, attribution_mask_processing, GradCAM
from models import VGGNB


# Load model and weights
model = VGGNB()
model.load_state_dict(torch.load('models\\best_VGGNB.pt', map_location=torch.device('cpu')))

# Define the required transformations for the VGGNB
img_size = 224
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size,img_size)),
            transforms.Normalize(
                mean=[0.36703529, 0.41083294, 0.50661294], std=[1, 1, 1])
        ])

# Load the desired image
path_img = 'Datasets\\Folds\\1\\Test\\ID343_UNIFESP_S32_pain.jpg'
target_class = 1 # 1-Pain / 0-No Pain
image = cv2.imread(path_img) # Load BGR image to VGGFace 
image_RGB = cv2.cvtColor(cv2.resize(image,(img_size,img_size)), cv2.COLOR_BGR2RGB) 
image = image/255 # Normalize to [0-1]
image = np.float32(image)
image = transform(image)

# Instatiate the Integrated Gradients and GradCam, for the GradCAM we are going
# to use the 'conv5_3' layer as target
ig = IntegratedGradients(model)
gradcam = GradCAM(model, model.VGGFace.features.conv5_3)

# Generate the attribution mask
heatmap_ig = ig.attribution_mask(image, target_class=target_class, n_steps=50)
heatmap_gradcam = gradcam.attribution_mask(image.unsqueeze(dim=0), target_class)

# Apply post-processing
result_ig, alpha_channel_ig = attribution_mask_processing(heatmap_ig)
result_cam, alpha_channel_cam = attribution_mask_processing(heatmap_gradcam)

# Define the plot colors
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"])

# Plot                                                    
plt.subplot(2,2,1)
plt.title('Integrated Gradients')
plt.imshow(heatmap_ig, cmap=cmap)
plt.imshow(image_RGB, alpha=0.5)
plt.axis('off')

plt.subplot(2,2,2)
plt.title('Integrated Gradients with Post-Processing')
plt.imshow(image_RGB)
plt.imshow(result_ig, cmap=cmap, alpha=alpha_channel_ig)
plt.axis('off')

plt.subplot(2,2,3)
plt.title('GradCAM')
plt.imshow(heatmap_gradcam, cmap=cmap)
plt.imshow(image_RGB, alpha=0.5)
plt.axis('off')

plt.subplot(2,2,4)
plt.title('GradCAM with Post-Processing')
plt.imshow(image_RGB)
plt.imshow(result_cam, cmap=cmap, alpha=alpha_channel_cam)
plt.axis('off')

plt.show()