import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TaskSpec:
    task_type: str  
    data_type: str  
    num_classes: Optional[int] = None
    input_shape: Optional[List[int]] = None
    target_column: Optional[str] = None

class Analyzer:
    def analyze(self, task_description: str, data_description: str) -> TaskSpec:
        task_type = self._infer_task_type(task_description)
        data_type = self._infer_data_type(data_description)
        num_classes = self._extract_num_classes(task_description)
        
        return TaskSpec(
            task_type=task_type,
            data_type=data_type,
            num_classes=num_classes
        )
    
    def _infer_task_type(self, text: str) -> str:
        text = text.lower()
        if any(x in text for x in ["classify", "classification", "categorize"]):
            return "classification"
        if any(x in text for x in ["predict", "regression", "estimate", "value"]):
            return "regression"
        if any(x in text for x in ["forecast", "time series", "future"]):
            return "forecasting"
        return "unknown"

    def _infer_data_type(self, text: str) -> str:
        text = text.lower()
        if any(x in text for x in ["image", "photo", "picture", "jpg", "png"]):
            return "image"
        if any(x in text for x in ["text", "document", "nlp", "sentence"]):
            return "text"
        if any(x in text for x in ["csv", "table", "excel", "dataframe", "tabular"]):
            return "tabular"
        return "unknown"

    def _extract_num_classes(self, text: str) -> Optional[int]:
        match = re.search(r"(\d+)\s+(classes|categories|labels)", text.lower())
        if match:
            return int(match.group(1))
        return None