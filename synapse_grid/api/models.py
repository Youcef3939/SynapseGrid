from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class TaskRequest(BaseModel):
    task_description: str
    data_description: str

class TaskResponse(BaseModel):
    task_type: str
    data_type: str
    num_classes: Optional[int] = None
    input_shape: Optional[List[int]] = None

class DesignRequest(BaseModel):
    task_spec: Dict[str, Any]
    compute_budget: str

class BlueprintResponse(BaseModel):
    name: str
    type: str
    params: Dict[str, Any]
    hpo_space: Optional[Dict[str, Any]] = None

class GenerateRequest(BaseModel):
    blueprint: Dict[str, Any]
    task_spec: Dict[str, Any]
    config: Dict[str, Any]

class CodeResponse(BaseModel):
    files: Dict[str, str]
