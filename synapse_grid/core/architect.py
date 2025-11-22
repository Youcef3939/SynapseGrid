from dataclasses import dataclass
from typing import Dict, Any, Optional
from .analyzer import TaskSpec

@dataclass
class ModelBlueprint:
    name: str
    type: str  
    params: Dict[str, Any]
    hpo_space: Optional[Dict[str, Any]] = None

class Architect:
    def design(self, task_spec: TaskSpec, compute_budget: str = "medium") -> ModelBlueprint:
        if task_spec.data_type == "image":
            return self._design_vision(task_spec, compute_budget)
        elif task_spec.data_type == "tabular":
            return self._design_tabular(task_spec, compute_budget)
        else:
            raise NotImplementedError(f"Architecture for {task_spec.data_type} not implemented yet.")

    def _design_vision(self, task_spec: TaskSpec, compute: str) -> ModelBlueprint:
        if compute == "low":
            return ModelBlueprint(
                name="mobilenet_v3_small",
                type="cnn",
                params={"num_classes": task_spec.num_classes, "pretrained": True},
                hpo_space={"lr": [1e-4, 1e-2], "batch_size": [32, 128]}
            )
        elif compute == "high":
            return ModelBlueprint(
                name="vit_b_16",
                type="transformer",
                params={"num_classes": task_spec.num_classes, "pretrained": True},
                hpo_space={"lr": [1e-5, 5e-4], "batch_size": [16, 64], "dropout": [0.0, 0.3]}
            )
        else:  
            return ModelBlueprint(
                name="resnet18",
                type="cnn",
                params={"num_classes": task_spec.num_classes, "pretrained": True},
                hpo_space={"lr": [1e-4, 1e-2], "batch_size": [32, 128]}
            )

    def _design_tabular(self, task_spec: TaskSpec, compute: str) -> ModelBlueprint:
        return ModelBlueprint(
            name="mlp",
            type="mlp",
            params={
                "input_dim": task_spec.input_shape[0] if task_spec.input_shape else 10, # Placeholder
                "output_dim": 1 if task_spec.task_type == "regression" else task_spec.num_classes,
                "hidden_dims": [128, 64] if compute == "low" else [512, 256, 128]
            },
            hpo_space={"lr": [1e-4, 1e-2], "dropout": [0.0, 0.5]}
        )