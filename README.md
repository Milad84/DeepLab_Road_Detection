# Satellite Image Segmentation for Road Extraction

## Libraries Used

![OpenCV](https://img.shields.io/badge/OpenCV-4.5.3.56-blue.svg) ![NumPy](https://img.shields.io/badge/NumPy-1.21.2-blue.svg) ![Pandas](https://img.shields.io/badge/Pandas-1.3.3-blue.svg) ![TQDM](https://img.shields.io/badge/TQDM-4.62.3-blue.svg) ![Seaborn](https://img.shields.io/badge/Seaborn-0.11.2-blue.svg) ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.3-blue.svg) ![Segmentation Models PyTorch](https://img.shields.io/badge/Segmentation_Models_PyTorch-0.2.0-blue.svg) ![Torch](https://img.shields.io/badge/Torch-1.9.0-blue.svg) ![Albumentations](https://img.shields.io/badge/Albumentations-1.0.3-blue.svg) ![GDAL](https://img.shields.io/badge/GDAL-3.3.3-blue.svg) 

![image](https://github.com/Milad84/DeepLab_Road_Detection/assets/38597478/e7daa473-d92a-4230-ac1b-ff631daf50a1)



This repository contains code for a satellite image segmentation project (https://www.kaggle.com/code/balraj98/road-extraction-from-satellite-images-deeplabv3) aimed at identifying road areas using deep learning techniques. The project involves training a segmentation model on satellite imagery data and making predictions on new satellite images.

## Overview

Satellite image segmentation is a computer vision task that involves partitioning an image into multiple segments or regions to simplify its representation. In this project, we focus on segmenting satellite images to identify road areas, which is crucial for various applications such as urban planning, transportation management, and infrastructure development.

## How is this different from the Kaggle post?

Although the code in many parts is inspired (copied) from the Kaggle post by BALRAJ ASHWATH, it differs in scope and scalability. What I am trying to do here is to train the model to be able to model other formats of satellite imagery and, to be specific, TIF. 


## How is the TIF file prepared to test the model?
The TIF samples are coming from [EarthExplorer.gov](https://earthexplorer.usgs.gov/)

### Major steps to prepare the target TIF and apply the model to it: 

Download satellite imagery from EarthExplorer.usgs.gov --> Split the Raster --> Use the model saved in the directory and inject your TIF --> Assign the coordinate system --> Save the output --> Convert to binary raster (0 and other value) where one value stands for the road, and the other value must be deleted --> Extract the needed value (Extract by Attribute) --> Vectorize the output --> Aggregate polygons (vectorized polygons).

The above process will give you a starting point for editing the width of the roads. The animated picture shows the difference before and after aggregation:

![3](https://github.com/Milad84/DeepLab_Road_Detection/assets/38597478/9b75031b-9ef7-412e-b97b-ec606d3ed2b6)




## Getting Started

### Installation

1. Clone this repository to your local machine or save the zipped folder:

   ```bash
   git clone https://github.com/your-username/DeepLab_Road_Detection.git

2. To train the model, you must also download the folder containing the "DeepGlobe Road Extraction Dataset" data. You can find the folder at https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset
## Usage

3. Preprocessing the satellite imagery data is essential before training the model. The downloaded folder contains the needed training pictures. However, detecting roads on your custom pictures involves tasks such as downloading, transforming, loading the data, applying necessary transformations, and organizing it into a format suitable for the model that will be applied.

### 1. Data Preprocessing

#### Steps:

1. **Import Required Libraries**: Ensure you have installed the necessary libraries. These may include OpenCV, NumPy, Pandas, and Albumentations.

2. **Load Data**: Load the satellite imagery data from the specified directory. This typically involves reading image files and corresponding mask files.

3. **Preprocess Data**: Apply preprocessing techniques such as resizing, normalization, and data augmentation to enhance the quality and diversity of the training data.

4. **Organize Data**: Organize the preprocessed data into training and validation sets. This may involve splitting the data into different directories or creating a DataFrame with file paths and labels.

### 2. Model Training

Once the data is preprocessed, the next step is to train the segmentation model using the prepared dataset. This section outlines the process of model training, including model selection, hyperparameter tuning, and optimization.

#### Steps:

1. **Import Required Libraries**: Ensure that you have the necessary libraries installed, including PyTorch, Segmentation Models PyTorch, and TorchVision.

2. **Define Model Architecture**: Choose a suitable segmentation model architecture (e.g., U-Net, DeepLabV3+) and configure it according to the requirements of the task.

3. **Configure Training Parameters**: Set the hyperparameters for model training, such as learning rate, batch size, number of epochs, and loss function.

4. **Train the Model**: Use the DataLoader utility to load the preprocessed data and train the segmentation model using the chosen architecture and training parameters.

5. **Monitor Training Progress**: Track the model's performance during training by monitoring metrics such as loss and accuracy on the training and validation sets.

6. **Save the Trained Model**: Once training is complete, save the trained model weights to disk for future use.

### 3. Inference on New Images

After training the model, you can use it to perform inference on new satellite images to segment road features. This section outlines the process of loading the trained model and applying it to new images.

#### Steps:

1. **Load Trained Model**: Load the saved model weights from disk using PyTorch or Segmentation Models PyTorch.

2. **Preprocess Input Image**: Preprocess the new satellite image to ensure compatibility with the trained model. This may involve resizing, normalization, and other transformations.

3. **Perform Inference**: Feed the preprocessed image into the trained model to generate segmentation masks for road features.

4. **Postprocess Output**: Post-process the segmentation mask to visualize the predicted road features and apply any necessary adjustments or refinements.

5. **Save Results**: Save the resulting segmentation mask or annotated image to disk for further analysis or visualization.

## Challenges and Considerations

Satellite image segmentation projects may encounter various challenges and considerations that impact model performance and results. Here are some key factors to keep in mind:

- **Data Quality**: Ensure that the satellite imagery data is of high quality and free from noise, artifacts, or distortions that could affect model performance.

- **Model Complexity**: Experiment with different segmentation model architectures, loss functions, and optimization techniques to find the best combination for the task at hand.

- **Computational Resources**: Consider the computational resources required for training deep learning models on large datasets, as well as the time and cost implications.

- **Evaluation Metrics**: Select appropriate evaluation metrics (e.g., IoU, Dice Coefficient) to assess the model's performance and compare different models objectively.

- **Deployment Considerations**: Think about how the trained model will be deployed in production environments and any additional considerations or requirements for deployment.

By addressing these challenges and considerations, you can develop effective solutions for satellite image segmentation tasks and contribute to various applications such as urban planning, environmental monitoring, and infrastructure development.





