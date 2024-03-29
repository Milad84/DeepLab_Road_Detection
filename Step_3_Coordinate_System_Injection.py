# You will use this code if you need to save the output somewhere on your system. 
#If saving the output is not a concern and you are just tuning your model, use Step_2.

import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from osgeo import gdal

# Define model parameters
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['background', 'road']
ACTIVATION = 'sigmoid'

# Define preprocessing function
def preprocess_image(image):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # Ensure input image is in RGB format
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # Pad the image to make its dimensions divisible by 16
    h, w, _ = image.shape
    new_h = int(np.ceil(h / 16) * 16)
    new_w = int(np.ceil(w / 16) * 16)
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    # Apply preprocessing function
    image = preprocessing_fn(image)
    return image.transpose(2, 0, 1).astype('float32')

# Specify the path to the model checkpoint file
model_checkpoint_path = 'best_model.pth'

# Check if the model checkpoint file exists
if not os.path.exists(model_checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint file '{model_checkpoint_path}' not found.")

# Load the model
model = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))  # Load the model on CPU

# Load and preprocess your single TIF file
input_image_path = r'G:\ATD\ACTIVE TRANS\Vision Zero\GIS\OnSystem-OffSystem\Sat Imagery\Pictures\Small Tiffs\3097_33_4_20150120_10.tif'

# Open the image using GDAL to retain geospatial information
ds = gdal.Open(input_image_path)
input_image = np.transpose(ds.ReadAsArray(), (1, 2, 0))

# Preprocess the input image
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

# Get the geotransform and projection from the input image
geotransform = ds.GetGeoTransform()
projection = ds.GetProjection()

# Save the output mask with geospatial information
output_path = r'G:\ATD\ACTIVE TRANS\Vision Zero\GIS\OnSystem-OffSystem\Sat Imagery\Pictures\Small Tiffs\Output\predicted_road_mask_with_GCS.tif'
driver = gdal.GetDriverByName('GTiff')
output_ds = driver.Create(output_path, road_mask.shape[1], road_mask.shape[0], 1, gdal.GDT_Byte)
output_ds.SetGeoTransform(geotransform)
output_ds.SetProjection(projection)
output_ds.GetRasterBand(1).WriteArray(road_mask)
output_ds = None
