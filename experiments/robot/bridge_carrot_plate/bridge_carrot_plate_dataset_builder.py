import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import logging
import h5py
from typing import Iterator, Tuple, Any, Dict, List

from experiments.robot.bridge_carrot_plate.dataset_builder import MultiThreadedDatasetBuilder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
N_VIEWS = 4
IMAGE_SIZE = (256, 256)
TRAIN_PROPORTION = 0.9


class BridgeCarrotPlateDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Bridge dataset subset with 'put carrot on plate' instructions."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_0": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation (fixed position).",
                                    ),
                                    "image_1": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position).",
                                    ),
                                    "image_2": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position)",
                                    ),
                                    "image_3": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "depth_0": tfds.features.Image(
                                        shape=IMAGE_SIZE + (1,),
                                        dtype=np.uint16,
                                        encoding_format="png",
                                        doc="Main camera depth observation (fixed position).",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot end effector state, consists of [3x XYZ, 3x roll-pitch-yaw, 1x gripper]",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [3x XYZ delta, 3x roll-pitch-yaw delta, 1x gripper absolute].",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                            "has_image_0": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image0 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_1": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image1 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_2": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image2 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_3": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image3 exists in observation, otherwise dummy value.",
                            ),
                            "has_depth_0": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if depth0 exists in observation, otherwise dummy value.",
                            ),
                            "has_language": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if language exists in observation, otherwise empty string.",
                            ),
                        }
                    ),
                }
            )
        )
    

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        data_root = "/slurm_home/byungjun_alinlab/openvla-oft/experiments/robot/bridge_carrot_plate/filtered_dataset"
        train_path = os.path.join(data_root, "put_carrot_on_plate_train_examples.hdf5")
        val_path = os.path.join(data_root, "put_carrot_on_plate_val_examples.hdf5")
        
        return {
            "train": self._generate_examples(train_path, "train"),
            "val": self._generate_examples(val_path, "val"),
        }
    
    def _generate_examples(self, path, split):
        """Yields examples from an HDF5 file."""
        # Open the HDF5 file
        with h5py.File(path, 'r') as h5_file:
            # Get all example groups
            example_keys = list(h5_file.keys())
            
            # Process each example
            for i, example_key in enumerate(example_keys):
                try:
                    example_group = h5_file[example_key]
                    
                    # Create the example dictionary
                    example = {
                        'steps': [],
                        'episode_metadata': {}
                    }
                    
                    # Process episode metadata
                    metadata_group = example_group['episode_metadata']

                    example['episode_metadata'] = {
                        'file_path': metadata_group['file_path'][()].decode('utf-8'),
                        'has_image_0': bool(metadata_group['has_image_0'][()]),
                        'has_image_1': bool(metadata_group['has_image_1'][()]),
                        'has_image_2': bool(metadata_group['has_image_2'][()]),
                        'has_image_3': bool(metadata_group['has_image_3'][()]),
                        'has_depth_0': False,
                        'has_language': bool(metadata_group['has_language'][()]),
                    }
                    
                    
                    # Process steps
                    steps_group = example_group['steps']
                    for step_key in sorted(steps_group.keys()):
                        step_group = steps_group[step_key]
                        step = {
                            'observation': {},
                            'is_first': bool(step_group['is_first'][()]),
                            'is_last': bool(step_group['is_last'][()]),
                            'language_instruction': step_group['language_instruction'][()].decode('utf-8')
                        }
                        
                        # Process observation
                        obs_group = step_group['observation']
                        for obs_key in obs_group.keys():
                            if obs_key.startswith('image_'):
                                # Get image array directly
                                img_array = obs_group[obs_key][()]
                                step['observation'][obs_key] = img_array
                            elif obs_key == 'depth_0':
                                # Get depth array directly
                                depth_array = obs_group[obs_key][()]
                                step['observation'][obs_key] = depth_array
                            elif obs_key == 'state':
                                # Get state as float32 array
                                state = np.array(obs_group[obs_key][()], dtype=np.float32)
                                step['observation'][obs_key] = state

                        if "depth_0" not in obs_group:
                            step['observation']['depth_0'] = np.zeros((256, 256, 1), dtype=np.uint16)

                        # Process action
                        if 'action' in step_group:
                            action = np.array(step_group['action'][()], dtype=np.float32)
                            step['action'] = action
                        
                        example['steps'].append(step)
                    
                    # Yield the example
                    yield f"{split}_{i}", example
                    
                except Exception as e:
                    logger.error(f"Error processing example {example_key}: {e}")
                    # Continue to the next example
                    continue
                    
                