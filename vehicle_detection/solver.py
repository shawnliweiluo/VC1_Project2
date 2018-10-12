import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from bbox_helper import generate_prior_bboxes, loc2bbox, center2corner, nms_bbox


class Solver:
    def __init__(self, train_data_set, valid_data_set, batch_size):
        self.train_data_set = train_data_set
        self.valid_data_set = valid_data_set
        self.batch_size = batch_size

        self.train_loader = DataLoader(self.train_data_set, batch_size=self.batch_size, shuffle=True, num_workers=6)
        self.valid_loader = DataLoader(self.valid_data_set, batch_size=self.batch_size, shuffle=True, num_workers=6)

        self.mean_img = np.asarray((127, 127, 127), dtype=np.float32).reshape(3, 1, 1)
        self.std_img = 128.0

    def evaluate_model(self, model, loss_criteria, loader=None, with_plots=False, prior_bboxes=None):
        # Put the model in evaluation mode
        model.eval()
        class_losses = []
        bbox_losses = []
        if loader is None:
            loader = self.valid_loader
        with torch.no_grad():
            for (x, gt_locs, gt_labels) in loader:
                # Move to the right device
                x = Variable(x.cuda())
                gt_locs = Variable(gt_locs.cuda())
                gt_labels = Variable(gt_labels.cuda())

                # Predict the classes and bbox coordinates for the input.
                confidences, locs = model.forward(x)

                # Compute the validation loss
                conf_loss, loc_huber_loss = loss_criteria.forward(confidences, locs, gt_labels, gt_locs)
                print(conf_loss , loc_huber_loss)
                class_losses.append(conf_loss.item())
                bbox_losses.append(loc_huber_loss.item())

            if with_plots:
                NUM_IMGS = 2
                fig, ax = plt.subplots(NUM_IMGS)
                img_sample = x[:NUM_IMGS].clone()
                loc_sample = locs[:NUM_IMGS, :, :].clone()
                conf_sample = confidences[:NUM_IMGS, :, :].clone()
                for i in range(1, NUM_IMGS + 1):
                    img = np.asarray(img_sample[i - 1])
                    img = (img * self.std_img) + self.mean_img
                    img = np.ascontiguousarray(img.transpose(1, 2, 0).astype('uint8'))
                    h, w, c = img.shape
                    # Convert the locations to bbox
                    bbox = loc2bbox(loc_sample[i - 1], prior_bboxes)
                    # Convert bbox to corner format
                    bbox = center2corner(bbox)
                    bbox[:, [0, 2]] *= w
                    bbox[:, [1, 3]] *= h
                    # Apply nms on the bounding boxes
                    sel_boxes = nms_bbox(bbox, conf_sample[i-1])
                    # Draw bounding boxes on the image
                    for (bbox, label) in sel_boxes:
                        (x, y, xw, yh) = bbox
                        rect = patches.Rectangle([x, y], xw - x, yh - y, linewidth=1, edgecolor='r',
                                                 facecolor='none')
                        ax[i - 1].add_patch(rect)
                    ax[i - 1].imshow(img)
                plt.show()

            return class_losses, bbox_losses


    def train(self, model, optimizer, loss_criteria, num_epochs=1, print_every=100):
        # Cache the loss history
        train_class_losses = []
        train_bbox_losses = []
        valid_class_losses = []
        valid_bbox_losses = []
        prior_bboxes = generate_prior_bboxes()
        for i in range(num_epochs):
            print("This is epoch %d" % i)
            for batch_idx, (x, gt_locs, gt_labels) in enumerate(self.train_loader):
                # set the model in training mode
                model.train()

                # Move to the right device
                x = Variable(x.cuda())
                gt_labels = Variable(gt_labels.cuda(), requires_grad=False)
                gt_locs = Variable(gt_locs.cuda(), requires_grad=False)

                # Compute the coordinates
                confidences, locs = model.forward(x)

                # Compute the loss and save it.
                conf_loss, loc_huber_loss = loss_criteria.forward(confidences, locs, gt_labels, gt_locs)
                total_loss = conf_loss + loc_huber_loss
                train_class_losses.append(conf_loss)
                train_bbox_losses.append(loc_huber_loss)

                # Zero out the gradients before optimization
                optimizer.zero_grad()

                # Backprop
                total_loss.backward()

                # Take an optimization step
                optimizer.step()

                if (batch_idx % print_every) == 0:
                    print("Epoch %d, iteration %d : loss is %.7f" % (i, batch_idx, total_loss.item()))
                    # valid_class_loss, valid_bbox_loss = self.evaluate_model(model, self.validation_loader, loss_criteria,
                                                                            # prior_bboxes=prior_bboxes)
                    # valid_class_losses += valid_class_loss
                    # valid_bbox_losses += valid_bbox_loss
            print('--------')
        return train_class_losses, train_bbox_losses, valid_class_losses, valid_bbox_losses
