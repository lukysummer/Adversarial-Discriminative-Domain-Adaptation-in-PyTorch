import numpy as np
import matplotlib.pyplot as plt
import time, os
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.utils import model_zoo

ImageFile.LOAD_TRUNCATED_IMAGES = True
use_cuda = torch.cuda.is_available()


def calculate_confusion_matrix(encoder, 
                               classifier, 
                               transform, # torchvision.transforms object
                               img_dir,   # name of directory containing images to be tested (to calculate confusion matrix)
                               classes,   # list containing class names
                               multi_label=True, # True for multi-label data, False for single-label data
                               threshold=0.3):   # detection threshold for multi-label data
  encoder.cuda().eval()
  classifier.cuda().eval()
  n_classes = len(classes)
  test_labels = []
  first_img = True
  
  # initialize confusion matrix 
  cm = np.zeros((n_classes, 3))
  for cl in os.listdir(img_dir): # for each class
    cls_i = classes.index(cl)
    for img_path in os.listdir(os.path.join(img_dir, cl)): # for each image
      test_labels.append(cls_i)
      img = Image.open(os.path.join(img_dir, cl, img_path)).convert('RGB')     
      img = transform(img)[:3, :, :].unsqueeze(0)    
      img = img.cuda() if use_cuda else img
      with torch.no_grad():
        logits = encoder(img)
        logits = torch.flatten(logits, 1)
        logits = classifier(logits).cpu().detach().numpy()
        
      if multi_label:  # Multi-label Prediction 
        logits = logits[0]
        pred = [1 if prob > threshold else 0 for i, prob in enumerate(logits)]
        # if all logits < threshold, predict the one with the largest logit
        if sum(pred)==0: 
          pred = [1 if i==np.argmax(logits) else 0 for i, prob in enumerate(logits)] 
        cm[cls_i] += pred
      else:
        preds = np.argmax(logits, axis=1)
        test_preds = preds if first_img else np.concatenate((test_preds, preds), axis=0)  
        first_img = False

  if multi_label==False:
    # using sklearn.metrics.confusion_matrix
    cm = confusion_matrix(test_labels, test_preds, labels=np.arange(n_classes))

  return cm
