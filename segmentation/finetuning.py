"""
Fine-tune Cellpose CPSAM model on custom dataset.

This script fine-tunes a pre-trained Cellpose CPSAM model on your custom dataset.
It expects images and corresponding masks in the dataset folder.

Expected dataset structure (everything in same directory):
    dataset/
        image1.jpg
        image1_masks.tif
        image2.jpg
        image2_masks.tif
        ...

Masks should be labeled masks where each object has a unique integer ID.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
from tqdm import tqdm
import time
import threading

try:
    from cellpose import models, io, train
    from cellpose.io import imread
except ImportError as e:
    raise ImportError(
        "Cellpose is not installed. Install with: pip install cellpose"
    ) from e


def load_training_data(
    dataset_dir: str,
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    mask_suffix: str = "_masks",
    mask_extensions: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg")
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load images and masks from dataset directory.
    
    Expected structure (everything in same directory):
    dataset/
        image1.jpg
        image1_masks.tif
        image2.jpg
        image2_masks.tif
        ...
    
    Args:
        dataset_dir: Directory containing both images and masks
        image_extensions: Valid image file extensions
        mask_suffix: Suffix pattern for mask files (default: "_masks")
        mask_extensions: Valid mask file extensions
        
    Returns:
        Tuple of (images, masks) lists where each is a numpy array
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory not found: {dataset_path}")
    
    # Find all image files (exclude mask files)
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(dataset_path.glob(f"*{ext}")))
        image_files.extend(list(dataset_path.glob(f"*{ext.upper()}")))
    
    # Filter out mask files
    image_files = [
        f for f in image_files 
        if not any(f.stem.endswith(suffix) for suffix in [mask_suffix, mask_suffix.replace("_", "")])
    ]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {dataset_path}")
    
    images = []
    masks = []
    
    print(f"Found {len(image_files)} potential image files...")
    print(f"Loading image-mask pairs from: {dataset_path}")
    
    for img_path in tqdm(sorted(image_files), desc="Loading data"):
        # Find corresponding mask file
        # Pattern: image_name.jpg -> image_name_masks.tif
        stem = img_path.stem
        mask_path = None
        
        # Try different mask extensions
        for ext in mask_extensions:
            # Try with _masks suffix
            candidate = dataset_path / f"{stem}{mask_suffix}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
            # Try uppercase extension
            candidate = dataset_path / f"{stem}{mask_suffix}{ext.upper()}"
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path is None or not mask_path.exists():
            print(f"Warning: No mask found for {img_path.name}, skipping...")
            continue
        
        try:
            # Load image
            img = imread(str(img_path))
            if img.ndim == 3 and img.shape[2] == 3:
                # Convert RGB to grayscale if needed
                img = img.mean(axis=2).astype(np.uint8)
            elif img.ndim == 3 and img.shape[2] == 4:
                # RGBA to grayscale
                img = img[:, :, :3].mean(axis=2).astype(np.uint8)
            
            # Load mask
            mask = imread(str(mask_path))
            if mask.ndim == 3:
                # Convert to grayscale if needed (take first channel)
                mask = mask[:, :, 0] if mask.shape[2] >= 1 else mask.mean(axis=2)
            
            # Ensure mask is integer type (labeled mask)
            mask = mask.astype(np.int32)
            
            # Validate dimensions match
            if img.shape[:2] != mask.shape[:2]:
                print(f"Warning: Shape mismatch for {img_path.name}: "
                      f"image {img.shape[:2]} vs mask {mask.shape[:2]}, skipping...")
                continue
            
            # Validate and relabel mask if needed
            mask_max = mask.max()
            mask_min = mask.min()
            img_height, img_width = img.shape[:2]
            
            # Check if mask has any objects (non-zero values)
            unique_labels = np.unique(mask)
            num_objects = len(unique_labels) - (1 if 0 in unique_labels else 0)
            if num_objects == 0:
                print(f"Warning: No objects found in mask for {img_path.name}, skipping...")
                continue
            
            # Check if mask values are suspiciously large (might be pixel coordinates)
            # Cellpose expects labeled masks with sequential IDs, typically < 10000 for most images
            if mask_max > 100000 or mask_max > (img_height * img_width * 0.1):
                print(f"Warning: Suspiciously large mask values for {img_path.name}: max={mask_max}")
                print(f"  Attempting to relabel mask...")
                # Try to relabel: create sequential IDs from 1 to num_objects
                mask_relabeled = np.zeros_like(mask, dtype=np.int32)
                unique_nonzero = unique_labels[unique_labels != 0]
                for new_id, old_id in enumerate(unique_nonzero, start=1):
                    mask_relabeled[mask == old_id] = new_id
                mask = mask_relabeled
                print(f"  Relabeled mask: {num_objects} objects with IDs 1-{num_objects}")
            
            # Final validation: ensure mask values are reasonable
            mask_max = mask.max()
            if mask_max < 0:
                print(f"Warning: Negative mask values for {img_path.name}, skipping...")
                continue
            
            images.append(img)
            masks.append(mask)
            
        except Exception as e:
            print(f"Error loading {img_path.name}: {e}, skipping...")
            continue
    
    if len(images) == 0:
        raise ValueError("No valid image-mask pairs found!")
    
    print(f"Successfully loaded {len(images)} image-mask pairs")
    return images, masks


def fine_tune_cellpose(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    base_model: str = "cpsam",
    model_name: str = "organoid_model",
    output_dir: str = "models",
    n_epochs: int = 100,
    learning_rate: float = 0.00001,
    batch_size: int = 1,
    nimg_per_epoch: Optional[int] = None,
    save_every: int = 10,
    use_gpu: Optional[bool] = None,
    channels: List[int] = [0, 0],  # [grayscale, grayscale]
    diameter: Optional[float] = None,
    weight_decay: float = 0.1,
) -> str:
    """
    Fine-tune a Cellpose CPSAM model on custom data.
    
    Args:
        images: List of training images (grayscale or RGB)
        masks: List of labeled masks (each object has unique integer ID)
        base_model: Base model to fine-tune from ("cpsam", "cyto2", "cyto", etc.)
        model_name: Name for the saved model
        output_dir: Directory to save the trained model
        n_epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        nimg_per_epoch: Number of images per epoch (None = use all)
        save_every: Save checkpoint every N epochs
        use_gpu: Whether to use GPU (None = auto-detect)
        channels: Channel configuration [grayscale/red, grayscale/red]
        diameter: Expected cell diameter in pixels (None = auto-estimate)
        weight_decay: Weight decay for regularization
        
    Returns:
        Path to the saved model
    """
    # Check GPU availability
    if use_gpu is None:
        from cellpose import core
        use_gpu = core.use_gpu()
        print(f"GPU available: {use_gpu}")
    
    # Create output directory
    output_path = Path(output_dir).resolve()  # Get absolute path
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Model will be saved to: {output_path / model_name}")
    
    # Initialize model from base model
    print(f"Loading base model: {base_model}")
    model = models.CellposeModel(
        gpu=use_gpu,
        model_type=base_model
    )
    
    # Prepare training data - ensure proper data types
    print("Preparing training data...")
    train_images = []
    train_masks = []
    
    for img, mask in zip(images, masks):
        # Ensure image is uint8
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        train_images.append(img)
        
        # Ensure mask is uint16 or int32 for labeled masks
        if mask.dtype != np.uint16 and mask.dtype != np.int32:
            mask = mask.astype(np.uint16)
        train_masks.append(mask)
    
    # Set nimg_per_epoch if not specified
    if nimg_per_epoch is None:
        nimg_per_epoch = len(images)
    
    # Train the model
    print(f"Starting training for {n_epochs} epochs...")
    print(f"Training parameters:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Images per epoch: {nimg_per_epoch}")
    print(f"  - Total training images: {len(images)}")
    print(f"  - Saving checkpoint every {save_every} epochs")
    print(f"\nNote: Training may take a long time (CPSAM models are large and slow).")
    print(f"Each epoch can take several minutes depending on image size and GPU.")
    print(f"Press Ctrl+C to interrupt (model will be saved if checkpoint exists).\n")
    sys.stdout.flush()
    
    start_time = time.time()
    training_active = threading.Event()
    training_active.set()
    
    # Thread to show periodic status updates and monitor checkpoints
    def status_monitor():
        """Print periodic status updates during training"""
        update_interval = 30  # Print every 30 seconds (more frequent)
        last_checkpoint_time = start_time
        checkpoint_files = []
        first_epoch_time = None
        
        while training_active.is_set():
            time.sleep(update_interval)
            if training_active.is_set():
                elapsed = time.time() - start_time
                
                # Check for checkpoint files (Cellpose may save with different names)
                checkpoint_dir = output_path
                if checkpoint_dir.exists():
                    # Look for any .pth or .pt files (PyTorch model files)
                    checkpoint_files_new = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
                    checkpoint_files_new = [f for f in checkpoint_files_new if model_name in f.name]
                    
                    if len(checkpoint_files_new) > len(checkpoint_files):
                        new_checkpoints = [f for f in checkpoint_files_new if f not in checkpoint_files]
                        for cp in new_checkpoints:
                            cp_time = cp.stat().st_mtime
                            if cp_time > last_checkpoint_time:
                                print(f"[Checkpoint] Saved: {cp.name} (at {elapsed/60:.1f} min)", flush=True)
                                last_checkpoint_time = cp_time
                                if first_epoch_time is None:
                                    first_epoch_time = elapsed / 60
                                    print(f"[Info] First epoch completed in {first_epoch_time:.1f} minutes", flush=True)
                        checkpoint_files = checkpoint_files_new
                
                # Estimate progress based on time
                if first_epoch_time:
                    # Use actual first epoch time to estimate
                    estimated_epochs = min(int(elapsed / 60 / first_epoch_time), n_epochs)
                else:
                    # Before first epoch completes, use rough estimate
                    estimated_epochs = min(int(elapsed / 60 / 4), n_epochs)
                
                progress_pct = min((estimated_epochs / n_epochs) * 100, 99) if n_epochs > 0 else 0
                
                # Check if main model file exists
                main_model_path = output_path / f"{model_name}"
                model_exists = main_model_path.exists()
                model_info = " (model file exists)" if model_exists else ""
                
                print(f"[Status] Epoch ~{estimated_epochs}/{n_epochs} (~{progress_pct:.0f}%), "
                      f"{elapsed/60:.1f} min elapsed{model_info}", flush=True)
    
    status_thread = threading.Thread(target=status_monitor, daemon=True)
    status_thread.start()
    
    # Use Cellpose's train function - expects train_data and train_labels as lists of arrays
    try:
        print("=" * 60)
        print("TRAINING STARTED - This may take a while...")
        print("=" * 60)
        sys.stdout.flush()
        
        # Check if train_seg accepts save_every parameter
        import inspect
        sig = inspect.signature(train.train_seg)
        
        train_kwargs = {
            "train_data": train_images,
            "train_labels": train_masks,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "model_name": model_name,
            "save_path": str(output_path),
        }
        if "save_every" in sig.parameters:
            train_kwargs["save_every"] = save_every
        
        train.train_seg(model.net, **train_kwargs)
        
        training_active.clear()  # Stop status monitor
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
        sys.stdout.flush()
        
    except KeyboardInterrupt:
        training_active.clear()  # Stop status monitor
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 60)
        print(f"Training ran for {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
        print("Checking for saved checkpoints...")
        sys.stdout.flush()
        
        # Check if a checkpoint was saved
        checkpoint_path = output_path / f"{model_name}"
        if checkpoint_path.exists():
            print(f"✓ Checkpoint found at: {checkpoint_path}")
            print("You can resume training or use this checkpoint.")
        else:
            print("✗ No checkpoint found. Training did not save progress.")
            print("Consider reducing --save-every to save more frequently.")
        sys.stdout.flush()
        raise
    except Exception as e:
        training_active.clear()  # Stop status monitor
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING ERROR")
        print("=" * 60)
        print(f"Error occurred after {elapsed_time/60:.1f} minutes: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    
    model_path = output_path / f"{model_name}"
    model_path_abs = model_path.resolve()
    print(f"\nTraining complete! Model saved to: {model_path_abs}")
    print(f"To use this model:")
    print(f"  from cellpose import models")
    print(f"  model = models.CellposeModel(gpu=True, pretrained_model='{model_path_abs}')")
    
    return str(model_path_abs)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Cellpose model on custom dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Path to dataset directory containing images and masks (default: dataset)"
    )
    parser.add_argument(
        "--mask-suffix",
        type=str,
        default="_masks",
        help="Suffix pattern for mask files (default: _masks)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="cpsam",
        choices=["cpsam", "cyto2", "cyto", "nuclei", "tissuenet", "livecell", "cyto3"],
        help="Base model to fine-tune from (default: cpsam)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="organoid_model",
        help="Name for the saved model (default: organoid_model)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model (default: models)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00001,
        help="Learning rate (default: 0.00001)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--nimg-per-epoch",
        type=int,
        default=8,
        help="Number of images per epoch (default: 8)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10)"
    )
    parser.add_argument(
        "--use-gpu",
        type=bool,
        default=True,
        help="Use GPU for training (default: True)"
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="Expected cell diameter in pixels (default: auto-estimate)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay for regularization (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Load training data
    print("=" * 60)
    print("Loading training data...")
    print("=" * 60)
    images, masks = load_training_data(
        dataset_dir=args.dataset,
        mask_suffix=args.mask_suffix
    )
    
    # Fine-tune model
    print("\n" + "=" * 60)
    print("Starting fine-tuning...")
    print("=" * 60)
    model_path = fine_tune_cellpose(
        images=images,
        masks=masks,
        base_model=args.base_model,
        model_name=args.model_name,
        output_dir=args.output_dir,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        nimg_per_epoch=args.nimg_per_epoch,
        save_every=args.save_every,
        use_gpu=args.use_gpu,
        diameter=args.diameter,
        weight_decay=args.weight_decay,
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)
    print("\nTo use the trained model:")
    print(f"  from cellpose import models")
    print(f"  model = models.CellposeModel(gpu=True, pretrained_model='{model_path}')")


if __name__ == "__main__":
    main()
