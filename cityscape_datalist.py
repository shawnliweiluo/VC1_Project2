import numpy as np
import json
import os
import random


class CityScapeDatalist:
    def __init__(self, base_img_dir, base_label_dir, img_size=(1024, 2048), dtype=np.float32):
        self.base_img_dir = base_img_dir
        self.base_label_dir = base_label_dir
        self.img_size = img_size
        self.dtype = dtype
        self.data_list = None
        self.train_list = None
        self.valid_list = None
        self.test_list = None

    def prepare_data_list(self):
        self.data_list = []
        for cur_dir, sub_dirs, files in os.walk(self.base_img_dir):
            for img_file in files:
                tokens = img_file.split('_')
                assert len(tokens) == 4
                city_name = tokens[0]
                img_number = tokens[1] + '_' + tokens[2]
                img_path = os.path.join(cur_dir, img_file)

                labels_path = os.path.join(self.base_label_dir, city_name,
                                           city_name + '_' + img_number + "_gtCoarse_polygons.json")
                labels = []
                with open(labels_path, 'r') as f:
                    frame_info = json.load(f)
                    for obj in frame_info["objects"]:
                        label = obj["label"]
                        polygons = np.asarray(obj["polygon"], dtype=np.float32)
                        left_top = np.min(polygons, axis=0)
                        right_bottom = np.max(polygons, axis=0)
                        labels.append({"label": label, "left_top": left_top, "right_bottom": right_bottom})
                self.data_list.append({"img_path": img_path, "labels": labels})

    def split_data(self, train_portion=0.7, valid_portion=0.1):
        random.shuffle(self.data_list)
        total_data_size = len(self.data_list)

        train_data_size = int(train_portion * total_data_size)
        valid_data_size = int(valid_portion * total_data_size)

        self.train_list = self.data_list[:train_data_size]
        self.valid_list = self.data_list[train_data_size: train_data_size + valid_data_size]
        self.test_list = self.data_list[train_data_size + valid_data_size:]
