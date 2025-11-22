from synapse_grid.generator.writer import Writer
import os

class Explainer:
    def __init__(self):
        self.writer = Writer()

    def generate_explanation_script(self, output_dir: str, context: dict):
        print("Generating XAI script...")
        
        explanation_context = context.copy()
        if "batch_size" not in explanation_context:
            explanation_context["batch_size"] = 32
        if "data_path" not in explanation_context:
            explanation_context["data_path"] = "./data"
            
        output_path = os.path.join(output_dir, "explain.py")
        self.writer.generate("explain.py.j2", explanation_context, output_path)