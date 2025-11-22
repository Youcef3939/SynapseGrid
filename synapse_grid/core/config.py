from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    project_name: str = "synapse_project"
    output_dir: str = "output"
    device: str = "cuda"  # or "cpu"
    seed: int = 42
    
    enable_hpo: bool = False
    n_trials: int = 20
    
    log_level: str = "INFO"
    use_tensorboard: bool = True

config = Config()