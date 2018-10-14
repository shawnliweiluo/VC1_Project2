import numpy as np
import torch.nn
from torch.utils.data import Dataset
from PIL import Image

from vehicle_detection.bbox_helper import generate_prior_bboxes, match_priors
from vehicle_detection.util.augmentations import *


class CityScapeDataset(Dataset):

    def __init__(self, dataset_list, input_dim=(300, 300), dtype=np.float32,
                 mode='train'):
        self.dataset_list = dataset_list
        self.input_dim = input_dim
        self.dtype=dtype
        self.mode=mode

        # TODO: implement prior bounding box
        self.prior_bboxes = generate_prior_bboxes()

        # Pre-process parameters:
        #  Normalize: (I-self.mean)/self.std
        self.mean = np.asarray((127, 127, 127), dtype=self.dtype).reshape(3,1,1)
        self.std = 128.0

    def resize(self, img, bbox):
        w, h = img.size
        img = img.resize(self.input_dim)
        w_ratio = self.input_dim[0] / float(w)
        h_ratio = self.input_dim[1] / float(h)
        bbox[:, [0, 2]] *= w_ratio
        bbox[:, [1, 3]] *= h_ratio
        return img, bbox

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes.
        :return img: input image as a tensor, dim: input_dim
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4)
        :return bbox_label: matched classification label, dim: (num_priors)
        """
        # TODO: implement data loading
        # Load image the image and it's bounding boxes
        img_path = self.dataset_list[idx]['img_path']
        img = Image.open(img_path)
        sample_bboxes = self.dataset_list[idx]['bounding_boxes']
        # Take the labels for the image
        sample_labels = self.dataset_list[idx]['labels']

        if self.mode == 'train':
            # Data Augmentation
            # First apply random cropping.
            img, sample_bboxes, sample_labels = RandomCrop(img, sample_bboxes, sample_labels)

            # TODO:: Alter the brightness randomly

        # Resize the image and it's bounding boxes to a fixed size
        img, sample_bboxes = self.resize(img, sample_bboxes)

        # Convert image to array and move the channels to first dim
        img = np.asarray(img, dtype=self.dtype).transpose(2, 0, 1)

        # Normalize the image with self.mean and self.std
        img = (img - self.mean) / self.std

        # Normalize the bounding box position value from 0 to 1
        sample_bboxes[:, [0,2]] /= self.input_dim[0]
        sample_bboxes[:, [1,3]] /= self.input_dim[1]

        # Convert image, bbox and label to tensor
        img = torch.from_numpy(img)
        sample_bboxes = torch.from_numpy(sample_bboxes)
        sample_labels = torch.from_numpy(sample_labels).long()

        # Do the matching prior and generate ground-truth labels as well as the boxes
        # The bbox_tensor will be in ssd location format.
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes, sample_bboxes, sample_labels,
                                                      iou_threshold=0.5)

        # Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box
        # [DEBUG] check the output.
        assert isinstance(bbox_label_tensor, torch.Tensor)
        assert isinstance(bbox_tensor, torch.Tensor)
        assert bbox_tensor.dim() == 2
        assert bbox_tensor.shape[1] == 4
        assert bbox_label_tensor.dim() == 1
        assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]

        return img, bbox_tensor, bbox_label_tensor.long()