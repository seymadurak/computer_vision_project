# Car Plate Recognition and Reconstruction with Deep Learning
## Overview
This repository contains two main Jupyter Notebook studies for automatic vehicle license plate detection, reconstruction, and character recognition using deep learning. The project is based on open-source data and includes both a basic pipeline and an improved version with hyperparameter optimization.

## Project Structure
Car Plate Recognition and Reconstruction with Deep Learning_V1.ipynb
First version: Implements end-to-end license plate detection and recognition using Faster R-CNN and EasyOCR, without hyperparameter tuning.

Car Plate Recognition and Reconstruction with Deep Learning_V2.ipynb
Second version: Improves the process by performing hyperparameter optimization (random search for learning rate and batch size), leading to better model performance.

## Dataset
The project uses the public License Plate Recognition dataset from Kaggle.
(https://www.kaggle.com/datasets/adilshamim8/license-plate-recognition)
How to use:
Download the dataset from Kaggle and upload it to your Google Drive.
In the notebooks, the dataset is read directly from the Drive path you specify.

## Requirements
Before running the notebooks, install the following packages:


!pip install easyocr
!pip install opencv-python-headless
Other core libraries used:
torch, torchvision, numpy, pandas, matplotlib, PIL (Pillow)

## Workflow
### V1 Notebook:
Reads images and bounding box annotations.
Trains a Faster R-CNN model for plate detection.
Crops the detected plate region and applies OCR with EasyOCR.
Visualizes results.

### V2 Notebook:
Performs random search for optimal hyperparameters (learning rate, batch size).
Trains the model using the best parameters.
Evaluates and visualizes improved results.

## Notes
Ensure your dataset path matches the location in your Drive.
Training requires a CUDA-enabled GPU for reasonable speed.
The EasyOCR and OpenCV (headless) packages must be installed for successful character recognition and image processing.


## Contact
For any questions or suggestions, please contact [durak.2222116@studenti.uniroma1.it].
