"""
Visualization Module

Generates GIFs, videos, and interactive Gradio demos.
"""

import os
import imageio
import numpy as np
import gradio as gr
from metadrive import MetaDriveEnv
from stable_baselines3 import PPO

# Fix: Import close_engine safely
try:
    from metadrive.engine.engineutils import close_engine
except ImportError:
    try:
        from metadrive.utils import close_engine
    except ImportError:
        def close_engine():
            print("⚠️ close_engine not found, skipping...")
            pass

from env_config import get_env_config

def generate_gif(model: PPO, seed: int, traffic_density: float, 
                 filename: str, num_steps: int = 600) -> str:
    """
    Generate a GIF of the policy running in a specific scenario.
    
    Args:
        model: Trained PPO model
        seed: Scenario seed
        traffic_density: Traffic density (0.0 - 1.0)
        filename: Output GIF path
        num_steps: Number of frames to generate
    
    Returns:
        Path to saved GIF
    """
    close_engine()
    
    # Configure environment for rendering
    env_config = get_env_config("train")
    env_config["start_seed"] = seed
    env_config["traffic_density"] = traffic_density
    env_config["use_render"] = True
    
    env = MetaDriveEnv(env_config)
    
    obs, _ = env.reset()
    frames = []
    
    print(f"Generating GIF: seed={seed}, density={traffic_density}...")
    
    for i in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Render top-down view
        frame = env.render(mode="top_down", screen_size=(500, 500))
        frames.append(frame)
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            break
    
    env.close()
    
    # Save GIF
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, frames, fps=10, loop=0)
    print(f"GIF saved: {filename}")
    
    return filename

def generate_demo_suite(model: PPO, output_dir: str = "outputs/demo") -> dict:
    """
    Generate a suite of GIFs for different scenarios.
    
    Args:
        model: Trained PPO model
        output_dir: Directory to save GIFs
    
    Returns:
        Dictionary mapping (seed, density) to filename
    """
    gif_index = {}
    
    seeds = [1008, 1010, 1012]
    densities = [0.05, 0.1, 0.2]
    
    for seed in seeds:
        for density in densities:
            filename = f"{output_dir}/seed{seed}_density{density}.gif"
            generate_gif(model, seed, density, filename)
            gif_index[(seed, density)] = filename
    
    return gif_index

def create_gradio_demo(gif_index: dict) -> gr.Blocks:
    """
    Create a Gradio interface to view pre-generated GIFs.
    
    Args:
        gif_index: Dictionary mapping (seed, density) to filename
    
    Returns:
        Gradio Blocks app
    """
    def show_gif(seed, traffic_density):
        key = (int(seed), float(traffic_density))
        return gif_index.get(key, None)
    
    with gr.Blocks(title="PPO MetaDrive Demo") as demo:
        gr.Markdown("# 🤖 PPO Highway Planner - Interactive Demo")
        gr.Markdown("Select a scenario seed and traffic density to view the policy rollout.")
        
        with gr.Row():
            seed_dropdown = gr.Dropdown(
                choices=[1008, 1010, 1012],
                value=1008,
                label="Scenario Seed"
            )
            density_dropdown = gr.Dropdown(
                choices=[0.05, 0.1, 0.2],
                value=0.1,
                label="Traffic Density"
            )
        
        gif_output = gr.Image(type="filepath", label="Simulation Rollout")
        
        run_button = gr.Button("View Rollout")
        run_button.click(
            fn=show_gif,
            inputs=[seed_dropdown, density_dropdown],
            outputs=gif_output
        )
    
    return demo
