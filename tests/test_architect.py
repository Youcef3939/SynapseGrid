import pytest
from synapse_grid.core.architect import Architect
from synapse_grid.core.analyzer import TaskSpec

def test_architect_vision_low_compute():
    architect = Architect()
    spec = TaskSpec(task_type="classification", data_type="image", num_classes=10)
    blueprint = architect.design(spec, compute_budget="low")
    
    assert blueprint.type == "cnn"
    assert "mobilenet" in blueprint.name

def test_architect_vision_high_compute():
    architect = Architect()
    spec = TaskSpec(task_type="classification", data_type="image", num_classes=100)
    blueprint = architect.design(spec, compute_budget="high")
    
    assert blueprint.type == "transformer"
    assert "vit" in blueprint.name

def test_architect_tabular():
    architect = Architect()
    spec = TaskSpec(task_type="regression", data_type="tabular", input_shape=[20])
    blueprint = architect.design(spec, compute_budget="medium")
    
    assert blueprint.type == "mlp"
    assert blueprint.params["input_dim"] == 20