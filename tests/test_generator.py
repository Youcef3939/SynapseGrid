import pytest
import os
import shutil
from synapse_grid.generator.writer import Writer

@pytest.fixture
def output_dir():
    path = "tests/output_test"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

def test_generator_project_structure(output_dir):
    writer = Writer()
    context = {
        "model_name": "resnet18",
        "model_type": "cnn",
        "num_classes": 10,
        "input_dim": 10,
        "hidden_dims": [64],
        "dropout": 0.1,
        "batch_size": 32,
        "lr": 0.001,
        "epochs": 1,
        "data_path": "dummy",
        "data_type": "image",
        "use_tensorboard": False,
        "enable_hpo": False,
        "hpo_space": None,
        "n_trials": 1
    }
    
    writer.generate_project(output_dir, context)
    
    assert os.path.exists(os.path.join(output_dir, "model.py"))
    assert os.path.exists(os.path.join(output_dir, "train.py"))
    assert os.path.exists(os.path.join(output_dir, "data"))