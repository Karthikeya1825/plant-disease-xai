"""
Configuration and Hyperparameters for Plant Disease Detection with XAI.
"""
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class ProjectConfig:
    # Random seed for full reproducibility
    seed: int = 42

    # Hardware & Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0  # 0 on Windows to avoid multiprocessing issues with shared memory
    pin_memory: bool = False

    # Image specifications
    image_size: int = 224
    image_channels: int = 3
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    # Dataset splits
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15

    # Directory Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data_dir: Path = project_root / "data" / "raw" / "PlantVillage-Dataset"
    processed_dir: Path = project_root / "data" / "processed"
    models_dir: Path = project_root / "models"
    reports_dir: Path = project_root / "reports"
    figures_dir: Path = reports_dir / "figures"
    app_dir: Path = project_root / "app"

    # CSV metadata paths
    train_csv: Path = processed_dir / "train.csv"
    val_csv: Path = processed_dir / "val.csv"
    test_csv: Path = processed_dir / "test.csv"
    classes_json: Path = processed_dir / "classes.json"
    metrics_json: Path = reports_dir / "metrics.json"

    # Training Hyperparameters
    batch_size: int = 32
    warmup_epochs: int = 3
    finetune_epochs: int = 10
    scratch_epochs: int = 12
    early_stopping_patience: int = 4

    warmup_lr: float = 1e-3
    finetune_lr: float = 1e-4
    scratch_lr: float = 1e-3
    weight_decay: float = 1e-4

    # XAI
    gradcam_target_layers: dict = None

    def __post_init__(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

config = ProjectConfig()
