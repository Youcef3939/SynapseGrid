import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from synapse_grid.core.analyzer import Analyzer
from synapse_grid.core.architect import Architect
from synapse_grid.generator.writer import Writer
from synapse_grid.core.config import config
import os
import tempfile 
 
st.set_page_config(page_title="SynapseGrid mission control", layout="wide")

st.title("🧠 SynapseGrid mission control")
st.markdown("### Autonomous Neural Architect System")

st.sidebar.header("Configuration")
compute_budget = st.sidebar.select_slider("compute budget", options=["low", "medium", "high"], value="medium")
enable_hpo = st.sidebar.checkbox("Enable HPO (Optuna)", value=False)
use_tensorboard = st.sidebar.checkbox("Enable TensorBoard", value=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Task definition")
    task_desc = st.text_area("describe your task", "classify images into 10 classes")
    data_desc = st.text_input("data path / description", "C:/data/images")
    
    if st.button("analyze & design"):
        with st.spinner("analyzing task..."):
            analyzer = Analyzer()
            task_spec = analyzer.analyze(task_desc, data_desc)
            st.session_state['task_spec'] = task_spec
            st.success(f"Task Detected: {task_spec.task_type} | Data: {task_spec.data_type}")
            
        with st.spinner("designing architecture..."):
            architect = Architect()
            blueprint = architect.design(task_spec, compute_budget)
            st.session_state['blueprint'] = blueprint
            st.success(f"architecture selected: {blueprint.name}")

with col2:
    st.subheader("blueprint visualization")
    if 'blueprint' in st.session_state:
        blueprint = st.session_state['blueprint']
        st.json({
            "name": blueprint.name,
            "type": blueprint.type,
            "Parameters": blueprint.params,
            "HPO Space": blueprint.hpo_space
        })

if 'blueprint' in st.session_state:
    st.markdown("---")
    st.subheader("Code Generation")
    if st.button("generate code"):
        with st.spinner("writing code..."):
            writer = Writer()
            output_dir = config.output_dir
            
            task_spec = st.session_state['task_spec']
            blueprint = st.session_state['blueprint']
            
            context = {
                "model_name": blueprint.name,
                "model_type": blueprint.type,
                "num_classes": task_spec.num_classes if task_spec.num_classes else 10,
                "input_dim": task_spec.input_shape[0] if task_spec.input_shape else 10,
                "hidden_dims": blueprint.params.get("hidden_dims", [128, 64]),
                "dropout": 0.1,
                "batch_size": 32,
                "lr": 0.001,
                "epochs": 10,
                "data_path": data_desc,
                "data_type": task_spec.data_type,
                "use_tensorboard": use_tensorboard,
                "enable_hpo": enable_hpo,
                "hpo_space": blueprint.hpo_space,
                "n_trials": config.n_trials
            }
            
            writer.generate_project(output_dir, context)
            st.success(f"code generated in {output_dir}")
        
            with open(os.path.join(output_dir, "model.py"), "r") as f:
                st.code(f.read(), language="python")
