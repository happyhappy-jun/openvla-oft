import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import logging
import argparse
import h5py
from PIL import Image
from typing import Tuple, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
N_VIEWS = 4
IMAGE_SIZE = (256, 256)
TRAIN_PROPORTION = 0.9


def create_filtered_dataset(
    filter_text="put carrot on plate", 
    output_dir="filtered_dataset", 
    max_examples=None,
    debug_mode=False
):
    """
    Create a filtered dataset with specific language instructions.
    
    Args:
        filter_text: Text to filter in language instructions
        output_dir: Directory to save the filtered dataset
        max_examples: Maximum number of examples to save per split (None for all examples)
        debug_mode: If True, only save 10 examples per split for debugging
    
    Returns:
        Tuple of paths to train and val HDF5 files
    """
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original dataset
    builder = tfds.builder("bridge_v2", data_dir="/shared/dataset")
    
    # Get dataset info to access sizes
    info = builder.info
    
    # Get the train split
    train_dataset = builder.as_dataset(split="train", shuffle_files=False)
    
    # Get the val split
    val_dataset = builder.as_dataset(split="val", shuffle_files=False)
    
    # Calculate total sizes (approximate if not available)
    try:
        train_size = info.splits['train'].num_examples
        val_size = info.splits['val'].num_examples
        logger.info(f"Dataset sizes - Train: {train_size}, Val: {val_size}")
    except:
        # If sizes are not available, we'll count manually
        logger.info("Dataset sizes not available from metadata, using estimates")
        train_size = 10000  # Use a reasonable estimate or calculate by iterating once
        val_size = 2000     # Use a reasonable estimate or calculate by iterating once
    
    # Define the filter function
    def contains_filter_text(sample):
        # Get the first step of the episode
        first_step = next(iter(sample["steps"]))
        # Check if language_instruction contains the filter text
        instruction = first_step["language_instruction"]
        return tf.strings.regex_full_match(tf.strings.lower(instruction), f".*{filter_text}.*")
    
    # Create HDF5 files
    train_output_path = os.path.join(output_dir, f"{filter_text.replace(' ', '_')}_train_examples.hdf5")
    val_output_path = os.path.join(output_dir, f"{filter_text.replace(' ', '_')}_val_examples.hdf5")
    
    # Process the train dataset with progress tracking
    train_count = 0
    checked_count = 0
    
    with h5py.File(train_output_path, 'w') as train_file:
        # Create a progress bar that shows percentage through dataset
        pbar = tqdm(total=train_size, desc=f"Processing train examples with '{filter_text}'")
        
        for example in train_dataset:
            checked_count += 1
            pbar.update(1)
            
            # Check if this example passes the filter
            first_step = next(iter(example["steps"]))
            instruction = first_step["language_instruction"].numpy().decode('utf-8')
            if filter_text.lower() not in instruction.lower():
                continue
                
            # Create a group for this example
            example_group = train_file.create_group(f"example_{train_count}")
            
            # Save episode metadata
            metadata_group = example_group.create_group("episode_metadata")
            for key, value in example["episode_metadata"].items():
                if isinstance(value, tf.Tensor):
                    if value.dtype == tf.string:
                        metadata_group.create_dataset(key, data=value.numpy(), dtype=h5py.special_dtype(vlen=str))
                    elif value.dtype == tf.bool:
                        metadata_group.create_dataset(key, data=bool(value.numpy()))
                    elif value.dtype == tf.int32:
                        metadata_group.create_dataset(key, data=int(value.numpy()))
            
            # Save steps
            steps_group = example_group.create_group("steps")
            for i, step in enumerate(example["steps"]):
                step_group = steps_group.create_group(f"step_{i}")
                
                # Save observation
                obs_group = step_group.create_group("observation")
                for obs_key, obs_value in step["observation"].items():
                    if obs_key.startswith("image_"):
                        # For images, save as JPEG
                        if isinstance(obs_value, tf.Tensor):
                            # Convert tensor to numpy array
                            img_array = obs_value.numpy()
                            # Save the image array directly
                            obs_group.create_dataset(obs_key, data=img_array, dtype=np.uint8)
                    elif obs_key == "depth_0":
                        # For depth images, save as PNG
                        if isinstance(obs_value, tf.Tensor):
                            # Convert tensor to numpy array
                            depth_array = obs_value.numpy()
                            # Save the depth array directly
                            obs_group.create_dataset(obs_key, data=depth_array, dtype=np.uint16)
                    elif obs_key == "state":
                        # For state, save as float32 array
                        if isinstance(obs_value, tf.Tensor):
                            obs_group.create_dataset(obs_key, data=obs_value.numpy().flatten(), dtype=np.float32)
                
                # Save action, is_first, is_last, and language_instruction
                if isinstance(step["action"], tf.Tensor):
                    step_group.create_dataset("action", data=step["action"].numpy().flatten(), dtype=np.float32)
                if isinstance(step["is_first"], tf.Tensor):
                    step_group.create_dataset("is_first", data=bool(step["is_first"].numpy()))
                if isinstance(step["is_last"], tf.Tensor):
                    step_group.create_dataset("is_last", data=bool(step["is_last"].numpy()))
                if isinstance(step["language_instruction"], tf.Tensor):
                    step_group.create_dataset("language_instruction", data=step["language_instruction"].numpy(), dtype=h5py.special_dtype(vlen=str))
            
            logger.info(f"Train example {train_count+1}: {instruction} (checked {checked_count}/{train_size} examples)")
            
            train_count += 1
            
            # Stop after saving max_examples if specified
            if max_examples is not None and train_count >= max_examples:
                logger.info(f"Reached maximum number of train examples ({max_examples}). Stopping after checking {checked_count}/{train_size} examples.")
                break
        
        pbar.close()
    
    # Similar process for validation set
    val_count = 0
    checked_count = 0
    
    with h5py.File(val_output_path, 'w') as val_file:
        pbar = tqdm(total=val_size, desc=f"Processing val examples with '{filter_text}'")
        
        for example in val_dataset:
            checked_count += 1
            pbar.update(1)
            
            # Check if this example passes the filter
            first_step = next(iter(example["steps"]))
            instruction = first_step["language_instruction"].numpy().decode('utf-8')
            if filter_text.lower() not in instruction.lower():
                continue
                
            # Create a group for this example
            example_group = val_file.create_group(f"example_{val_count}")
            
            # Save episode metadata
            metadata_group = example_group.create_group("episode_metadata")
            for key, value in example["episode_metadata"].items():
                if isinstance(value, tf.Tensor):
                    if value.dtype == tf.string:
                        metadata_group.create_dataset(key, data=value.numpy(), dtype=h5py.special_dtype(vlen=str))
                    elif value.dtype == tf.bool:
                        metadata_group.create_dataset(key, data=bool(value.numpy()))
                    elif value.dtype == tf.int32:
                        metadata_group.create_dataset(key, data=int(value.numpy()))
            
            # Save steps
            steps_group = example_group.create_group("steps")
            for i, step in enumerate(example["steps"]):
                step_group = steps_group.create_group(f"step_{i}")
                
                # Save observation
                obs_group = step_group.create_group("observation")
                for obs_key, obs_value in step["observation"].items():
                    if obs_key.startswith("image_"):
                        # For images, save as JPEG
                        if isinstance(obs_value, tf.Tensor):
                            # Convert tensor to numpy array
                            img_array = obs_value.numpy()
                            # Save the image array directly
                            obs_group.create_dataset(obs_key, data=img_array, dtype=np.uint8)
                    elif obs_key == "depth_0":
                        # For depth images, save as PNG
                        if isinstance(obs_value, tf.Tensor):
                            # Convert tensor to numpy array
                            depth_array = obs_value.numpy()
                            # Save the depth array directly
                            obs_group.create_dataset(obs_key, data=depth_array, dtype=np.uint16)
                    elif obs_key == "state":
                        # For state, save as float32 array
                        if isinstance(obs_value, tf.Tensor):
                            obs_group.create_dataset(obs_key, data=obs_value.numpy().flatten(), dtype=np.float32)
                
                # Save action, is_first, is_last, and language_instruction
                if isinstance(step["action"], tf.Tensor):
                    step_group.create_dataset("action", data=step["action"].numpy().flatten(), dtype=np.float32)
                if isinstance(step["is_first"], tf.Tensor):
                    step_group.create_dataset("is_first", data=bool(step["is_first"].numpy()))
                if isinstance(step["is_last"], tf.Tensor):
                    step_group.create_dataset("is_last", data=bool(step["is_last"].numpy()))
                if isinstance(step["language_instruction"], tf.Tensor):
                    step_group.create_dataset("language_instruction", data=step["language_instruction"].numpy(), dtype=h5py.special_dtype(vlen=str))
            
            logger.info(f"Val example {val_count+1}: {instruction} (checked {checked_count}/{val_size} examples)")
            
            val_count += 1
            
            # Stop after saving max_examples if specified
            if max_examples is not None and val_count >= max_examples:
                logger.info(f"Reached maximum number of val examples ({max_examples}). Stopping after checking {checked_count}/{val_size} examples.")
                break
                
        pbar.close()
    
    logger.info(f"Found and saved {train_count} train examples (checked {checked_count}/{train_size}) and {val_count} val examples containing '{filter_text}'")
    logger.info(f"Train examples saved to {train_output_path}")
    logger.info(f"Val examples saved to {val_output_path}")
    
    return train_output_path, val_output_path


def main():
    """Main function to parse arguments and run the filtering script."""
    parser = argparse.ArgumentParser(description="Filter Bridge dataset for specific language instructions")
    parser.add_argument("--filter_text", type=str, default="put carrot on plate", 
                        help="Text to filter in language instructions")
    parser.add_argument("--output_dir", type=str, default="filtered_dataset", 
                        help="Directory to save the filtered dataset")
    parser.add_argument("--max_examples", type=int, default=None, 
                        help="Maximum number of examples to save per split (None for all examples)")
    parser.add_argument("--debug", action="store_true", 
                        help="Debug mode: only save 10 examples per split")
    parser.add_argument("--full", action="store_true", 
                        help="Process the full dataset (no limit on examples)")
    
    args = parser.parse_args()
    
    # Handle conflicting options
    if args.debug and args.full:
        logger.warning("Both --debug and --full options specified. --full takes precedence.")
        args.debug = False
    
    # Set max_examples based on options
    if args.debug:
        args.max_examples = 10
        logger.info("Debug mode enabled: only saving 10 examples per split")
    elif args.full:
        args.max_examples = None
        logger.info("Full dataset mode enabled: processing all examples")
    elif args.max_examples is None:
        # Default to debug mode if no options specified
        args.max_examples = 3
        logger.info("No mode specified, defaulting to 3 examples per split")
    
    # Create the filtered dataset
    train_path, val_path = create_filtered_dataset(
        filter_text=args.filter_text,
        output_dir=args.output_dir,
        max_examples=args.max_examples
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 