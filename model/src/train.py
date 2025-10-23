"""Training script with model versioning and registry management.

This script trains MLP models, saves them with version control,
and maintains a model registry for MLOps best practices.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional
from mlp import MLP


# Define patterns (same as notebook)
# fmt: off
PATTERNS = {
    "b": np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ], dtype=np.uint8),
    "d": np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ], dtype=np.uint8),
    "f": np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ], dtype=np.uint8),
}
# fmt: on

SEED = 42
RNG = np.random.default_rng(SEED)


def generate_sample(pattern: Literal["b", "d", "f"], noise: float = 0.0) -> np.ndarray:
    """
    Generates a 1D array based on a given pattern letter, with optional noise.
    
    Args:
        pattern: One of 'b', 'd', or 'f'.
        noise: Proportion of pixels to flip (0-1).
    Returns:
        1D array representing the 10x10 matrix.
    """
    sample = PATTERNS[pattern].copy()
    num_pixels = sample.size
    num_noisy = int(noise * num_pixels / 2)

    if num_noisy > 0:
        indices = RNG.choice(num_pixels, num_noisy, replace=False)
        sample[indices] = 1 - sample[indices]

    return sample


def generate_dataset(n_samples: int) -> pd.DataFrame:
    """
    Generates a dataset of pattern samples.
    10% samples have zero noise, 90% have noise between 0.01 and 0.30.
    
    Args:
        n_samples: Number of samples to generate.
    Returns:
        DataFrame with 100 columns for pattern and 'class' column.
    """
    columns = [str(i) for i in range(100)] + ["class"]
    df = pd.DataFrame(0, index=np.arange(n_samples), columns=columns)
    df = df.astype({"class": "str"})

    for i in range(n_samples):
        if i < int(0.1 * n_samples):
            noise = 0.0
        else:
            noise = RNG.uniform(0.01, 0.30)

        pattern = RNG.choice(list(PATTERNS.keys()))
        sample = generate_sample(pattern, noise).flatten()
        df.iloc[i, :100] = sample
        df.loc[i, "class"] = pattern

    return df


def split_dataset(
    df: pd.DataFrame, validation_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into training and validation sets.
    
    Args:
        df: Dataset to split.
        validation_ratio: Proportion for validation set.
    Returns:
        Tuple of (training_df, validation_df).
    """
    shuffled_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    validation_size = int(len(df) * validation_ratio)
    validation_df = shuffled_df.iloc[:validation_size].reset_index(drop=True)
    training_df = shuffled_df.iloc[validation_size:].reset_index(drop=True)
    return training_df, validation_df


def get_model_registry_path() -> Path:
    """Get path to model registry file."""
    return Path(__file__).parent.parent / "trained_models" / "model_registry.json"


def load_model_registry() -> dict:
    """Load model registry from disk."""
    registry_path = get_model_registry_path()
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            return json.load(f)
    return {"models": []}


def save_model_registry(registry: dict) -> None:
    """Save model registry to disk."""
    registry_path = get_model_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"📋 Registry updated at {registry_path}")


def register_model(
    model_path: Path,
    version: str,
    accuracy: float,
    hyperparameters: dict,
    dataset_info: dict,
    is_production: bool = False
) -> None:
    """
    Register a trained model in the model registry.
    
    Args:
        model_path: Path to the saved model file.
        version: Model version string.
        accuracy: Validation accuracy.
        hyperparameters: Model hyperparameters.
        dataset_info: Information about training dataset.
        is_production: Whether this is the production model.
    """
    registry = load_model_registry()
    
    # Mark all other models as non-production if this is production
    if is_production:
        for model in registry["models"]:
            model["is_production"] = False
    
    model_entry = {
        "version": version,
        "filename": model_path.name,
        "created_at": datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "hyperparameters": hyperparameters,
        "dataset_info": dataset_info,
        "is_production": is_production,
    }
    
    registry["models"].append(model_entry)
    save_model_registry(registry)


def train_and_save_model(
    n_samples: int = 1000,
    validation_ratio: float = 0.2,
    activation_type: Literal["sigmoid", "linear"] = "sigmoid",
    learning_rate: float = 0.1,
    momentum: float = 0.1,
    tolerance: float = 1e-6,
    version: Optional[str] = None,
    is_production: bool = False,
) -> Path:
    """
    Train a new MLP model and save it with versioning.
    
    Args:
        n_samples: Number of samples in dataset.
        validation_ratio: Proportion for validation.
        activation_type: Activation function type.
        learning_rate: Learning rate for training.
        momentum: Momentum coefficient.
        tolerance: Training stopping tolerance.
        version: Model version (auto-generated if None).
        is_production: Mark as production model.
    Returns:
        Path to saved model file.
    """
    print("=" * 60)
    print("🚀 TRAINING NEW MLP MODEL")
    print("=" * 60)
    
    # Generate dataset
    print(f"\n📊 Generating dataset with {n_samples} samples...")
    full_dataset = generate_dataset(n_samples)
    train_data, val_data = split_dataset(full_dataset, validation_ratio)
    print(f"   Train: {len(train_data)} samples | Validation: {len(val_data)} samples")
    
    # Initialize and train model
    print(f"\n🧠 Initializing MLP...")
    print(f"   Activation: {activation_type}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Momentum: {momentum}")
    
    mlp = MLP(
        activation_type=activation_type,
        learning_rate=learning_rate,
        momentum=momentum,
        seed=SEED,
    )
    
    print(f"\n🏋️  Training model (tolerance={tolerance})...")
    mlp.train(train_data, val_data, tolerance=tolerance, verbose=True)
    
    # Evaluate
    print(f"\n📈 Evaluating model...")
    eval_results = mlp.evaluate(val_data)
    accuracy = eval_results["accuracy"]
    print(f"   Accuracy: {accuracy:.4f} ({eval_results['correct']}/{eval_results['total']})")
    
    # Generate version string
    if version is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v1.0_{timestamp}_acc{accuracy:.3f}"
    
    # Save model
    models_dir = Path(__file__).parent.parent / "trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"mlp_{version}.pkl"
    
    print(f"\n💾 Saving model...")
    mlp.save(model_path)
    
    # Register in registry
    register_model(
        model_path=model_path,
        version=version,
        accuracy=accuracy,
        hyperparameters={
            "activation_type": activation_type,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "tolerance": tolerance,
        },
        dataset_info={
            "n_samples": n_samples,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "validation_ratio": validation_ratio,
        },
        is_production=is_production,
    )
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE")
    print(f"   Version: {version}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Production: {'Yes' if is_production else 'No'}")
    print("=" * 60)
    
    return model_path


def get_production_model() -> Optional[MLP]:
    """
    Load the current production model from registry.
    
    Returns:
        Loaded MLP model or None if no production model exists.
    """
    registry = load_model_registry()
    
    for model_entry in registry["models"]:
        if model_entry.get("is_production", False):
            models_dir = Path(__file__).parent.parent / "trained_models"
            model_path = models_dir / model_entry["filename"]
            if model_path.exists():
                return MLP.load(model_path)
    
    return None


if __name__ == "__main__":
    # Train a production-ready model
    train_and_save_model(
        n_samples=1000,
        validation_ratio=0.2,
        activation_type="sigmoid",
        learning_rate=0.1,
        momentum=0.1,
        tolerance=1e-6,
        is_production=True,
    )
