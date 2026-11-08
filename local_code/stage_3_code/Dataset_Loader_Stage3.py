"""Load one of the professor-provided Stage 3 pickle datasets."""

import pickle
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader_Stage3(dataset):
    """Return image tensors as NCHW arrays and labels as integer arrays."""

    DATASET_SETTINGS = {
        'mnist': {'file_name': 'MNIST', 'channels': 1, 'classes': 10, 'label_offset': 0},
        'orl': {'file_name': 'ORL', 'channels': 1, 'classes': 40, 'label_offset': 1},
        'cifar': {'file_name': 'CIFAR', 'channels': 3, 'classes': 10, 'label_offset': 0},
        'cifar10': {'file_name': 'CIFAR', 'channels': 3, 'classes': 10, 'label_offset': 0},
    }

    def __init__(self, dataset_name, data_folder=None):
        key = dataset_name.lower()
        if key not in self.DATASET_SETTINGS:
            raise ValueError("dataset_name must be 'mnist', 'orl', or 'cifar'")

        self.dataset_key = key
        self.settings = self.DATASET_SETTINGS[key]
        super().__init__(key.upper(), key.upper() + ' image classification')
        self.data_folder = Path(data_folder) if data_folder else (
            Path(__file__).resolve().parents[2] / 'data' / 'stage_3_data'
        )

    def _prepare_split(self, instances, split_name):
        images = []
        labels = []
        expected_channels = self.settings['channels']

        for index, instance in enumerate(instances):
            image = np.asarray(instance['image'])
            label = int(instance['label']) - self.settings['label_offset']

            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            if image.ndim != 3:
                raise ValueError(
                    '{} example {} has unsupported image shape {}'.format(
                        split_name, index, image.shape
                    )
                )
            # The professor's images are height x width x channels.
            if image.shape[-1] not in (1, 3) and image.shape[0] in (1, 3):
                image = np.transpose(image, (1, 2, 0))
            if self.dataset_key == 'orl' and image.shape[-1] == 3:
                # The three ORL channels are identical; use the first channel.
                image = image[:, :, :1]
            if image.shape[-1] != expected_channels:
                raise ValueError(
                    '{} example {} has {} channels; expected {}'.format(
                        split_name, index, image.shape[-1], expected_channels
                    )
                )

            images.append(np.transpose(image, (2, 0, 1)))
            labels.append(label)

        if not images:
            raise ValueError('{} split is empty'.format(split_name))

        # Float32 and [0, 1] scaling match common PyTorch image input practice.
        X = np.stack(images).astype(np.float32, copy=False)
        if X.max() > 1.0:
            X /= 255.0
        y = np.asarray(labels, dtype=np.int64)

        if y.min() < 0 or y.max() >= self.settings['classes']:
            raise ValueError('{} labels are outside the configured class range'.format(split_name))
        return {'X': X, 'y': y}

    def load(self):
        """Load the existing train/test lists without changing their partition."""
        source_file = self.data_folder / self.settings['file_name']
        if not source_file.is_file():
            raise FileNotFoundError('Stage 3 dataset file not found: {}'.format(source_file))

        with source_file.open('rb') as file:
            loaded = pickle.load(file)

        if 'train' not in loaded or 'test' not in loaded:
            raise ValueError('{} must contain train and test lists'.format(source_file))

        return {
            'train': self._prepare_split(loaded['train'], 'train'),
            'test': self._prepare_split(loaded['test'], 'test'),
            'classes': self.settings['classes'],
            'channels': self.settings['channels'],
        }
