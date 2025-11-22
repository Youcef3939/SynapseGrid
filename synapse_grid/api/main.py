from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from synapse_grid.core.analyzer import Analyzer, TaskSpec
from synapse_grid.core.architect import Architect
from synapse_grid.generator.writer import Writer
from synapse_grid.api.models import TaskRequest, TaskResponse, DesignRequest, BlueprintResponse, GenerateRequest, CodeResponse
import os
import tempfile

app = FastAPI(title="SynapseGrid API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=TaskResponse)
async def analyze_task(request: TaskRequest):
    analyzer = Analyzer()
    spec = analyzer.analyze(request.task_description, request.data_description)
    return TaskResponse(
        task_type=spec.task_type,
        data_type=spec.data_type,
        num_classes=spec.num_classes,
        input_shape=spec.input_shape
    )

@app.post("/design", response_model=BlueprintResponse)
async def design_architecture(request: DesignRequest):
    architect = Architect()
    spec_dict = request.task_spec
    spec = TaskSpec(
        task_type=spec_dict.get("task_type"),
        data_type=spec_dict.get("data_type"),
        num_classes=spec_dict.get("num_classes"),
        input_shape=spec_dict.get("input_shape")
    )
    
    blueprint = architect.design(spec, request.compute_budget)
    return BlueprintResponse(
        name=blueprint.name,
        type=blueprint.type,
        params=blueprint.params,
        hpo_space=blueprint.hpo_space
    )

@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: GenerateRequest):
    writer = Writer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        context = {
            "model_name": request.blueprint.get("name"),
            "model_type": request.blueprint.get("type"),
            "num_classes": request.task_spec.get("num_classes", 10),
            "input_dim": 10,
            "hidden_dims": request.blueprint.get("params", {}).get("hidden_dims", [128, 64]),
            "dropout": 0.1,
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 10,
            "data_path": "data",
            "data_type": request.task_spec.get("data_type"),
            "use_tensorboard": True,
            "enable_hpo": False,
            "hpo_space": request.blueprint.get("hpo_space"),
            "n_trials": 10
        }
        
        writer.generate_project(temp_dir, context)
        
        files = {}
        for filename in ["model.py", "train.py"]:
            path = os.path.join(temp_dir, filename)
            if os.path.exists(path):
                with open(path, "r") as f:
                    files[filename] = f.read()
                    
        return CodeResponse(files=files)

@app.get("/health")
def health():
    return {"status": "online", "system": "SynapseGrid"}