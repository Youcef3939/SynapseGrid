import os
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any

class Writer:
    def __init__(self, template_dir: str = "synapse_grid/templates"):
        if not os.path.isabs(template_dir):
             base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
             template_dir = os.path.join(base_dir, "templates")
        
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def generate(self, template_name: str, context: Dict[str, Any], output_path: str):
        template = self.env.get_template(template_name)
        content = template.render(**context)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Generated {output_path}")

    def generate_project(self, output_dir: str, context: Dict[str, Any]):
        self.generate("model.py.j2", context, os.path.join(output_dir, "model.py"))
        self.generate("train.py.j2", context, os.path.join(output_dir, "train.py"))
        
        if context.get("enable_hpo"):
            self.generate("hpo.py.j2", context, os.path.join(output_dir, "hpo.py"))
            
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)