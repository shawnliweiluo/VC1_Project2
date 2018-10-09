import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from cityscape_dataset import *
from bbox_loss import *


class Solver:
    def __init__(self, train_data_set, valid_data_set, batch_size, device=torch.device('cuda')):
        self.train_data_set = train_data_set
        self.valid_data_set = valid_data_set
        self.batch_size = batch_size

        self.train_loader = DataLoader(self.train_data_set, batch_size=self.batch_size, shuffle=True, num_workers=6)
        self.valid_loader = DataLoader(self.valid_data_set, batch_size=self.batch_size, shuffle=True, num_workers=6)


    def compute_accuracy(self, model, loader, loss_criteria, radius=np.array([5, 10, 15, 20])):
        # Put the model in evaluation mode
        model.eval()
        num_samples = 0
        num_detected = np.zeros(len(radius))
        validation_loss = []
        with torch.no_grad():
            for (x, y) in loader:
                # Move to the right device
                x = Variable(x.cuda())
                y = Variable(y.cuda())

                # Compute the land mark coordinates.
                predictions = model.forward(x)

                # Compute the validation loss
                loss = loss_criteria(predictions, y)
                validation_loss.append(loss.item())

                # Find the distance to true label
                dist = (predictions - y) ** 2

                # Combine the x and y distances
                dist = np.add.reduceat(dist, torch.arange(dist.shape[1])[::2], axis=1)
                dist = dist.sqrt()
                N, num_landmarks = dist.shape
                num_samples += (N * num_landmarks)
                # Compute number of correctly identified landmarks
                # for each radius.
                for i in range(len(radius)):
                    num_detected[i] += (dist < radius[i]).sum()
            accuracy = 100 * (num_detected / float(num_samples))
            accuracy = dict(zip(radius, accuracy))
            return validation_loss, accuracy


    def train(self, model, optimizer, loss_function, num_epochs=1, print_every=100, radius=[1, 2], radius_eval_idx=0):
        train_class_loss, train_bbox_loss, valid_class_loss, valid_bbox_loss = [], [], [], []
        for i in range(num_epochs):
            print("This is epoch %d" % i)
            for batch_idx, (img_input, gt_bbox_locs, gt_bbox_labels) in enumerate(self.train_loader):
                # set the model in training mode
                model.train()

                # Move to the right device
                img_input = Variable(img_input.cuda())
                gt_bbox_locs = Variable(gt_bbox_locs.cuda())
                gt_bbox_labels = Variable(gt_bbox_labels.cuda())

                # Compute the coordinates
                confidences, bbox_locs = model.forward(img_input)

                # Compute the loss and save it.
                conf_loss, loc_huber_loss = loss_function.forward(confidences, bbox_locs, gt_bbox_labels, gt_bbox_locs)

                train_class_loss.append(conf_loss)
                train_bbox_loss.append(loc_huber_loss)

                # Zero out the gradients before optimization
                optimizer.zero_grad()

                # Backprop
                loss_function.backward()

                # Take an optimization step
                optimizer.step()

                # if (batch_idx % print_every) == 0:
                    # print("Epoch %d, iteration %d : loss is %.7f" % (i, t, loss.item()))
                    # Define a range of radius values for accuracy computations.
                    # loss_t, accuracy = self.ComputeAccuracy(model, self.validation_loader, loss_criteria, radius=radius)
                    # validation_loss += loss_t
                    # Print the accuracy on some radius
                    # print("The accuracy on validation set for radius {} is: {}".format(radius[radius_eval_idx],
                    #                                                                    accuracy[radius[radius_eval_idx]]))
            print('--------')
        return train_class_loss, train_bbox_loss, valid_class_loss, valid_bbox_loss#, accuracy
