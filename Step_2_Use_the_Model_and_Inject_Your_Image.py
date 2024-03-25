import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as album

# Define model parameters
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['background', 'road']
ACTIVATION = 'sigmoid'

# Define preprocessing function
def preprocess_image(image):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    image = preprocessing_fn(image)
    return image.transpose(2, 0, 1).astype('float32')

# Load the model
model = torch.load('best_model.pth', map_location=torch.device('cpu'))  # Load the model on CPU

# Load and preprocess your single TIF file
input_image_path = r'G:\ATD\ACTIVE TRANS\Vision Zero\GIS\OnSystem-OffSystem\Sat Imagery\Pictures\Small Tiffs\3097_33_4_20150120_10.tif'
input_image = cv2.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Pad the input image to make dimensions divisible by 16
h, w, _ = input_image.shape
new_h = int(np.ceil(h / 16) * 16)
new_w = int(np.ceil(w / 16) * 16)
pad_top = (new_h - h) // 2
pad_bottom = new_h - h - pad_top
pad_left = (new_w - w) // 2
pad_right = new_w - w - pad_left
input_image = cv2.copyMakeBorder(input_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

input_image = preprocess_image(input_image)

# Perform inference
with torch.no_grad():
    input_tensor = torch.from_numpy(input_image).unsqueeze(0)
    model.eval()
    output = model(input_tensor)

# Process the output as needed
output_mask = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
predicted_class_index = np.argmax(output_mask, axis=0)  # Get the index of the class with the highest probability

# Assuming road class is class 1, create binary mask for road class
road_mask = (predicted_class_index == 1).astype(np.uint8) * 255

# Save the output mask
output_path = r'G:\ATD\ACTIVE TRANS\Vision Zero\GIS\OnSystem-OffSystem\Sat Imagery\Pictures\Small Tiffs\Output\predicted_road_mask.png'
cv2.imwrite(output_path, road_mask)  # Save the road mask as an image
