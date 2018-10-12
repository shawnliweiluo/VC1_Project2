import numpy as np
import torch.nn
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

from ssd_net import *
from bbox_loss import *

from cityscape_datalist import *
from cityscape_dataset import *
from solver import *


# Set default figure size
plt.rcParams['figure.figsize'] = (30.0, 40.0)

# Set image and label directory
imgs_dir = "/home/rasad/Documents/Project02/cityscapes_samples"
labels_dir = "/home/rasad/Documents/Project02/cityscapes_samples_labels"

# Set accepted labels for this project
label_dict = {
    "person": 1,
    "persongroup": 2,
    "rider": 3,
    "bicycle": 4,
    "bicyclegroup": 5,
    "car": 6,
    "cargroup": 7,
    "bus": 8,
    "truck": 9,
    "traffic sign": 10,
    "traffic light": 11
}

# Create a data list of images
cs_data_list = CityScapeDatalist(imgs_dir, labels_dir, label_dict)
cs_data_list.prepare_data_list()

# Split data into train test and validation
cs_data_list.split_data()

# Create a dataset using the data list you prepared
input_dim=(300, 300)
train_dataset = CityScapeDataset(cs_data_list.train_list, input_dim=input_dim)
valid_dataset = CityScapeDataset(cs_data_list.valid_list, input_dim=input_dim)

# Create a solver to train the model
solver = Solver(train_dataset, valid_dataset, batch_size=32)

# Instantiate the model
num_classes = len(set(label_dict.values())) + 1
ssd_net = SSD(num_classes).cuda()

# Create an optimizer
optimizer = torch.optim.Adam(ssd_net.parameters(), lr=1e-3)

# Initialize loss function
loss_function = MultiboxLoss()

train_bbox_losses = []
train_class_losses = []

# Training the model
lr = 1e-3
for num_epochs in [45, 15]:
	optimizer.param_groups[0]['lr'] = lr
	tclass_loss, tbbox_loss, vclass_loss, vbbox_loss = solver.train(ssd_net, 
                                                                    optimizer, 
                                                                    loss_function,
                                                                    num_epochs=num_epochs,
                                                                    print_every=10)
	train_class_losses += tclass_loss
	train_bbox_losses += tbbox_loss
	lr *= 1e-1

# Save the model
net_state = ssd_net.state_dict()
torch.save(net_state, 'ssd_net')


