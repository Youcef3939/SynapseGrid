import typer
from rich.console import Console
from rich.panel import Panel
from synapse_grid import __version__

app = typer.Typer(
    name="synapse-grid",
    help="Autonomous Neural Architect System",
    add_completion=False,
)
console = Console()

@app.command()
def build(
    task: str = typer.Option(..., help="Description of the task (e.g., 'classify images')"),
    data: str = typer.Option(..., help="Path to data or description"),
    compute: str = typer.Option("medium", help="Compute budget (low, medium, high)"),
    hpo: bool = typer.Option(False, help="Enable Hyperparameter Optimization"),
):
    from synapse_grid.core.analyzer import Analyzer
    from synapse_grid.core.architect import Architect
    from synapse_grid.generator.writer import Writer
    from synapse_grid.core.config import config
    import os

    console.print(Panel.fit(f"[bold blue]SynapseGrid v{__version__}[/bold blue]\n[green]Analyzing task: {task}[/green]"))
    
    analyzer = Analyzer()
    with console.status("[bold green]Analyzing requirements...[/bold green]"):
        task_spec = analyzer.analyze(task, data)
    console.print(f"[bold]Task Detected:[/bold] {task_spec.task_type} | [bold]Data:[/bold] {task_spec.data_type}")
    
    architect = Architect()
    with console.status("[bold blue]Designing neural architecture...[/bold blue]"):
        blueprint = architect.design(task_spec, compute)
    console.print(f"[bold]Architecture Selected:[/bold] {blueprint.name} ({blueprint.type})")
    
    writer = Writer()
    output_dir = config.output_dir
    
    context = {
        "model_name": blueprint.name,
        "model_type": blueprint.type,
        "num_classes": task_spec.num_classes if task_spec.num_classes else 10, # default
        "input_dim": task_spec.input_shape[0] if task_spec.input_shape else 10,
        "hidden_dims": blueprint.params.get("hidden_dims", [128, 64]),
        "dropout": 0.1, # default
        "batch_size": 32, # default
        "lr": 0.001, # default
        "epochs": 10,
        "data_path": data,
        "data_type": task_spec.data_type,
        "use_tensorboard": config.use_tensorboard,
        "enable_hpo": hpo,
        "hpo_space": blueprint.hpo_space,
        "n_trials": config.n_trials
    }
    
    with console.status("[bold magenta]Generating code...[/bold magenta]"):
        writer.generate_project(output_dir, context)
        
    console.print(f"[bold green]Build Complete![/bold green] Code generated in [underline]{output_dir}[/underline]")


@app.command()
def deploy(
    model_path: str = typer.Option(..., help="Path to the trained model file"),
):
    from synapse_grid.deploy.deployer import Deployer
    import os
    
    console.print(f"[bold green]Deploying model from {model_path}...[/bold green]")
    
    deployer = Deployer()
    output_dir = os.path.dirname(model_path) 
    
    context = {
        "model_name": "resnet18", 
        "model_type": "cnn",
        "num_classes": 10,
    }
    
    with console.status("[bold cyan]Generating deployment package...[/bold cyan]"):
        deployer.generate_deployment(output_dir, context)
        
    console.print(f"[bold green]Deployment ready![/bold green]\nRun: [yellow]cd {output_dir} && docker build -t synapse-model . && docker run -p 8000:8000 synapse-model[/yellow]")


if __name__ == "__main__":
    app()