import pytest
from synapse_grid.core.analyzer import Analyzer

def test_analyzer_classification():
    analyzer = Analyzer()
    spec = analyzer.analyze("classify images into 10 classes", "folder of jpgs")
    assert spec.task_type == "classification"
    assert spec.data_type == "image"
    assert spec.num_classes == 10

def test_analyzer_regression():
    analyzer = Analyzer()
    spec = analyzer.analyze("predict house prices", "csv file")
    assert spec.task_type == "regression"
    assert spec.data_type == "tabular"

def test_analyzer_unknown():
    analyzer = Analyzer()
    spec = analyzer.analyze("do something magic", "unknown data")
    assert spec.task_type == "unknown"