from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import json
import argparse

# Args for train.py
def get_input_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="path to images", default="flowers/", choices=['flowers/', 'flowers/valid', 'flowers/test'])
    parser.add_argument("--arch", type=str, help="vgg or densenet", default="vgg", choices=['vgg', 'densenet'])
    parser.add_argument('--hidden_units', type=int) #1024 for vgg and 500 for densenet
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--gpu', type=str, default='cuda')
    parser.add_argument('--category file', type=str, default='cat_to_name.json')
    args = parser.parse_args()

    return args

# Args for predict.py
def get_input_args_predict():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to images", default='flowers/train/1/image_06734.jpg')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument('--gpu', type=str, default='cuda')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json')
    args = parser.parse_args()

    return args

# Function that loads labels
def load_category_names(category_file):
    with open(category_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
        
# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    epoch = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    criterion = checkpoint['criterion']
    
    if checkpoint['arch'] == 'vgg':
        model = models.vgg16(pretrained=True)    
    elif checkpoint['arch'] == 'densenet':
        model = models.densenet121(pretrained=True)
    
    #model = models.arch(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Function that takes an image, peforms operations and returns np.array
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    size = (256, 256)
    
    img = Image.open(image)
    
    # Resize image
    img = img.resize(size)
    
    # Retrieve values to be used in crop center
    left_mrgn = (img.width - 224) / 2
    bottom_mrgn = (img.height - 224) / 2
    right_mrgn = left_mrgn + 224
    top_mrgn = bottom_mrgn + 224
    
    # Crop center portion of image
    img = img.crop((left_mrgn, bottom_mrgn, right_mrgn, top_mrgn))
    
    # Color channels
    img = np.array(img) / 255
    
    # Mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    
    # Normalize
    img = (img - mean) / std_dev
    
    # Transpose
    np_img = img.transpose((2, 0, 1))
    
    return np_img

def predict(image_path, model, device_p, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    idx_to_class = dict()
    top_label = list()
    top_prob_1 = list()
    
    # Retrieve image
    image = process_image(image_path)
    
    # Process image
    # Convert type
    # Add batch of 1 and retrieve probabilities
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.unsqueeze_(0)

    if device_p == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model.to(torch.device('cuda:0'))
        
    probs = torch.exp(model.forward(image.cuda()))
    
    # Get top probabilities with associated indices
    top_5_p, top_5_l = probs.topk(topk)
    top_prob = top_5_p.cpu().detach().numpy().tolist()[0]
    top_lab = top_5_l.cpu().detach().numpy().tolist()[0]
    
    # Invert and convert
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key
    
    for label in top_lab:
        top_label.append(int(idx_to_class[label]))
    
    return top_prob, top_label

# Generates visual of image and bar chart of top 5 probable flower classes
def print_results(image, model, device, cat_to_name, top_k):
    # Create list
    classes = list()

    # Run prediction
    probs, flowers = predict(image, model, device, top_k)

    # Prints the most likely image class and it's associated probability
    # Prints out the top K classes along with associated probabilities
    for i in range(len(flowers)):
        if str(flowers[i]) in cat_to_name:
            print('Flower Class: {} Probability: {}'.format(cat_to_name[str(flowers[i])], probs[i]))
            