import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import glob
from tqdm import tqdm

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity

# import helper
from misc import toimage

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None):

        is_aug=False
        if year=='2012_aug':
            is_aug = True
            year = '2012'

        self.root = os.path.expanduser(root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform

        self.image_set = image_set
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if is_aug and image_set=='train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            split_f = os.path.join( self.root, 'train_aug.txt')#'./datasets/data/train_aug.txt'
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]

        self.voc_root = voc_root
        self.setup_annotations()
        self.masks = [os.path.join(mask_dir, 'pre_encoded', x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        # return 10
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

    def setup_annotations(self):
        """Sets up Berkley annotations by adding image indices to the
        `train_aug` split and pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). This
        function also defines the `train_aug` and `train_aug_val` data splits
        according to the description in the class docstring
        """
        target_path = os.path.join(self.voc_root, "SegmentationClass/pre_encoded")
        #target_path = "pre_encoded"
        # print(target_path)
        # exit()
        os.makedirs(target_path,exist_ok=True)
        pre_encoded = glob.glob(os.path.join(target_path, "*.png"))
        expected = np.unique(self.images).size

        if len(pre_encoded) != expected:
            print("Pre-encoding segmentation masks...")

            for ii in tqdm(self.masks):
                fname = os.path.basename(ii) #+ ".png"
                # print(fname)
                # exit()
                lbl_path = os.path.join(self.voc_root, "SegmentationClass", fname)
                if os.path.exists(f'{target_path}/{fname}'):
                    # print(f'{target_path}/{fname}')
                    continue
                #lbl = self.encode_segmap(m.imread(lbl_path))
                lbl = self.encode_segmap(Image.open(lbl_path).convert('RGB'))
                lbl = toimage(lbl, high=lbl.max(), low=lbl.min())
                #m.imsave(pjoin(target_path, fname), lbl)
                #pdb.set_trace()
                aa = np.array(lbl)
                # if np.max(np.unique(aa)) > 20:
                # 	pdb.set_trace()
                lbl.save(f'{target_path}/{fname}')

				#break

		# assert expected == 9733, "unexpected dataset sizes"

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

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
        	mask (np.ndarray): raw segmentation label image of dimension
        	  (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
        	(np.ndarray): class map with dimensions (M,N), where the value at
        	a given location is the integer denoting the class index.
        """
        mask = np.array(mask)
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)

        #pdb.set_trace()
        for ii, label in enumerate(self.get_pascal_labels()):
        	label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        	#print(ii)

        label_mask = label_mask.astype(int)

        #pdb.set_trace()
        if np.max(np.unique(label_mask)) > 20:
        	pdb.set_trace()

        return label_mask

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

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)
