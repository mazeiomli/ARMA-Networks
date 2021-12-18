import os
import os.path
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']

    image_dir = os.path.join(data_root, 'JPEGImages')
    mask_dir = os.path.join(data_root, 'SegmentationClass')
    splits_dir = os.path.join(data_root, 'ImageSets/Segmentation')
    split_f = os.path.join(splits_dir, split.rstrip('\n') + '.txt')

    if not os.path.exists(split_f):
        raise ValueError(
            'Wrong image_set entered! Please use image_set="train" '
            'or image_set="trainval" or image_set="val"')

    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]

    image_list = [os.path.join(image_dir, x + ".jpg") for x in file_names]
    label_list = [os.path.join(mask_dir, "pre_encoded", x + ".png") for x in file_names]
    assert (len(image_list) == len(label_list))

    image_label_list = [(image_name, label_name) for image_name,label_name in zip(image_list,label_list)]

    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = Image.open(image_path)
        image = np.float32(np.asarray(image))
        label = np.asarray(Image.open(label_path))
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
        	np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
			[
				[0, 0, 0],
				[128, 0, 0],
				[0, 128, 0],
				[128, 128, 0],
				[0, 0, 128],
				[128, 0, 128],
				[0, 128, 128],
				[128, 128, 128],
				[64, 0, 0],
				[192, 0, 0],
				[64, 128, 0],
				[192, 128, 0],
				[64, 0, 128],
				[192, 0, 128],
				[64, 128, 128],
				[192, 128, 128],
				[0, 64, 0],
				[128, 64, 0],
				[0, 192, 0],
				[128, 192, 0],
				[0, 64, 128],
			]
		)

    def decode_segmap(self, label_mask, plot=False):
    	"""Decode segmentation class labels into a color image
    	Args:
    		label_mask (np.ndarray): an (M,N) array of integer values denoting
    		  the class label at each spatial location.
    		plot (bool, optional): whether to show the resulting color image
    		  in a figure.
    	Returns:
    		(np.ndarray, optional): the resulting decoded color image.
    	"""
    	label_colours = self.get_pascal_labels()
    	r = label_mask.copy()
    	g = label_mask.copy()
    	b = label_mask.copy()
    	for ll in range(0, self.n_classes):
    		r[label_mask == ll] = label_colours[ll, 0]
    		g[label_mask == ll] = label_colours[ll, 1]
    		b[label_mask == ll] = label_colours[ll, 2]
    	rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    	rgb[:, :, 0] = r / 255.0
    	rgb[:, :, 1] = g / 255.0
    	rgb[:, :, 2] = b / 255.0
    	if plot:
    		plt.imshow(rgb)
    		plt.show()
    	else:
    		return rgb
