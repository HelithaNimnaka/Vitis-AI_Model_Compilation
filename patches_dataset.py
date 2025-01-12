import numpy as np
import cv2
from pathlib import Path
import torch
import torch.utils.data as data
from utils.tools import dict_update

class PatchesDataset(data.Dataset):
    default_config = {
        'dataset': 'hpatches',
        'alteration': 'all',
        'cache_in_memory': False,
        'truncate': None,
        'preprocessing': {
            'resize': False
        }
    }

    def __init__(self, transform=None, **config):
        self.config = dict_update(self.default_config, config)
        self.files = self._init_dataset(**self.config)
        sequence_set = []
        for (img, img_warped, mat_hom) in zip(self.files['image_paths'], self.files['warped_image_paths'], self.files['homography']):
            sample = {'image': img, 'warped_image': img_warped, 'homography': mat_hom}
            sequence_set.append(sample)
        self.samples = sequence_set
        self.transform = transform
        if 'resize' in config['preprocessing'] and config['preprocessing']['resize']:
            self.sizer = np.array(config['preprocessing']['resize'])
        else:
            self.sizer = None

    def __getitem__(self, index):
        def _read_image(path):
            print(f"Attempting to read image at: {path}")  # Debugging statement
            input_image = cv2.imread(path)
            if input_image is None:
                print(f"Failed to read image at: {path}")  # Debugging message for missing/corrupt files
                raise FileNotFoundError(f"Image file not found or corrupt: {path}")
            return input_image

        def _preprocess(image):
            if self.sizer is not None:
                s = max(self.sizer / image.shape[:2])
                image = cv2.resize(image, (self.sizer[1], self.sizer[0]), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image.astype('float32') / 255.0
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            if self.transform is not None:
                image = self.transform(image)
            return image

        sample = self.samples[index]
        image_original = _read_image(sample['image'])
        image = _preprocess(image_original)
        warped_image = _preprocess(_read_image(sample['warped_image']))
        homography = sample['homography']

        return {'image': image, 'warped_image': warped_image, 'homography': homography}

    def __len__(self):
        return len(self.samples)

    def _init_dataset(self, **config):
        dataset_folder = 'COCO/patches' if config['dataset'] == 'coco' else 'HPatches'
        base_path = Path(config['path'], dataset_folder)
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths, warped_image_paths, homographies = [], [], []

        for path in folder_paths:
            if config['alteration'] == 'i' and path.stem[0] != 'i':
                continue
            if config['alteration'] == 'v' and path.stem[0] != 'v':
                continue

            num_images = 1 if config['dataset'] == 'coco' else 5
            file_ext = '.ppm' if config['dataset'] == 'hpatches' else '.jpg'

            for i in range(2, 2 + num_images):
                img_path = Path(path, "1" + file_ext)
                warped_img_path = Path(path, str(i) + file_ext)
                homography_path = Path(path, f"H_1_{i}")

                # Debugging statements
                if not img_path.exists():
                    print(f"Missing original image: {img_path}")
                if not warped_img_path.exists():
                    print(f"Missing warped image: {warped_img_path}")
                if not homography_path.exists():
                    print(f"Missing homography file: {homography_path}")

                # Add paths if they exist
                if img_path.exists() and warped_img_path.exists() and homography_path.exists():
                    image_paths.append(str(img_path))
                    warped_image_paths.append(str(warped_img_path))
                    homographies.append(np.loadtxt(homography_path))

        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
            warped_image_paths = warped_image_paths[:config['truncate']]
            homographies = homographies[:config['truncate']]

        return {'image_paths': image_paths, 'warped_image_paths': warped_image_paths, 'homography': homographies}
