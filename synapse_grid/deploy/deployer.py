import os
from synapse_grid.generator.writer import Writer

class Deployer:
    def __init__(self):
        self.writer = Writer()

    def generate_deployment(self, output_dir: str, context: dict):
        print("Generating deployment files...")
        self.writer.generate("serve.py.j2", context, os.path.join(output_dir, "serve.py"))
        self.writer.generate("Dockerfile.j2", context, os.path.join(output_dir, "Dockerfile"))
        
        with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
            f.write("torch\ntorchvision\nfastapi\nuvicorn\npillow\n")