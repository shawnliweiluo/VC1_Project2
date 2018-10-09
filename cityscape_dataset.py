import numpy as np
import torch.nn
from torch.utils.data import Dataset
from bbox_helper import generate_prior_bboxes, match_priors
from PIL import Image
import matplotlib.pyplot as plt

imgs_dir = "cityscapes_samples"
labels_dir = "cityscapes_samples_labels"

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


class CityScapeDataset(Dataset):

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

        # TODO: implement prior bounding box
        # self.prior_bboxes = generate_prior_bboxes(prior_layer_cfg='Todo, use your own setting, please refer bbox_helper.py for an example')

        # Pre-process parameters:
        #  Normalize: (I-self.mean)/self.std
        self.mean = np.asarray((127, 127, 127))
        self.std = 128.0

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4)
        :return bbox_label: matched classification label, dim: (num_priors)
        """

        # 1. Load image as well as the bounding box with its label
        # 2. Normalize the image with self.mean and self.std
        # 3. Convert the bounding box from corner form (left-top, right-bottom): [(x,y), (x+w, y+h)] to
        #    center form: [(center_x, center_y, w, h)]
        # 4. Normalize the bounding box position value from 0 to 1
        sample_labels = []
        sample_bboxes = []

        data = self.dataset_list[idx]
        img_path, labels = data["img_path"], data["objects"]

        sample_img = Image.open(img_path)
        sample_img = np.asarray(sample_img, dtype=np.float32)
        sample_img = (sample_img - sample_img.mean()) / sample_img.std()
        h, w, c = sample_img.shape[0], sample_img.shape[1], sample_img.shape[2]
        img_tensor = torch.from_numpy(sample_img)
        img_tensor = img_tensor.view(c, h, w)

        for item in labels:
            label, left_top, right_bottom = item["label"], item["left_top"], item["right_bottom"]
            # ['vegetation', 'sky', 'pole', 'fence', 'sidewalk', 'car', 'person', 'traffic sign', 'building', 'road',
            # 'parking', 'ego vehicle', 'rectification border', 'out of roi', 'bicyclegroup', 'bicycle', 'persongroup',
            # 'rider', 'cargroup', 'wall', 'traffic light', 'truck', 'terrain', 'motorcycle', 'rail track', 'bus', 'train']
            if label not in label_dict:
                sample_labels.append(0)
            else:
                sample_labels.append(label_dict[label])

            bb_w, bb_h = right_bottom[0] - left_top[0], right_bottom[1] - left_top[1]
            bb_center_x, bb_center_y = left_top[0] + bb_w / 2.0, left_top[1] + bb_h / 2.0
            bb_w, bb_h, bb_center_x, bb_center_y = bb_w / w, bb_h / h, bb_center_x / w, bb_center_y / h
            sample_bboxes.append((bb_center_x, bb_center_y, bb_w, bb_h))

        sample_bboxes = torch.from_numpy(np.asarray(sample_bboxes))
        sample_labels = torch.from_numpy(np.asarray(sample_labels))

        # 4. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box

        # 5. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes, sample_bboxes, sample_labels, iou_threshold=0.5)

        # [DEBUG] check the output.
        assert isinstance(bbox_label_tensor, torch.Tensor)
        assert isinstance(bbox_tensor, torch.Tensor)
        assert bbox_tensor.dim() == 2
        assert bbox_tensor.shape[1] == 4
        assert bbox_label_tensor.dim() == 1
        assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]

        return img_tensor, bbox_tensor, bbox_label_tensor