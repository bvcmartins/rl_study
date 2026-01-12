# COMPREHENSIVE LOGGING AND VISUALIZATION UTILITIES FOR PONG DQN CNN
# This module contains all logging, visualization, and tracking functions for training monitoring
# Compatible with both original (32-64-64) and v2 optimized (32-36-20) CNN architectures

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import wandb
import os
from pathlib import Path


# =============================================================================
# BASIC LOGGING FUNCTIONS
# =============================================================================

def log_hyperparameters(artifacts_dir, architecture_info=None):
    """Log hyperparameters to MLflow with optional architecture info"""
    params = {
        "artifacts_dir": artifacts_dir
    }

    # Add architecture-specific parameters if provided
    if architecture_info:
        # Filter out string values that could cause wandb issues
        filtered_info = {}
        for key, value in architecture_info.items():
            # Only log numeric or boolean values to wandb-safe params
            if isinstance(value, (int, float, bool)):
                filtered_info[key] = value
        params.update(filtered_info)

    mlflow.log_params(params)


def log_model_info(model, device, architecture_name="CNN-DQN"):
    """Log comprehensive model information to both MLflow and wandb"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate parameter breakdown by layer
    conv_params = sum(p.numel() for name, p in model.named_parameters() if 'conv' in name)
    fc_params = sum(p.numel() for name, p in model.named_parameters() if 'fc' in name)
    
    # Calculate model size in MB
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    model_metrics = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "conv_parameters": conv_params,
        "fc_parameters": fc_params,
        "model_size_mb": model_size_mb
    }
    
    # Log to MLflow as parameters
    mlflow.log_params(model_metrics)
    
    # Log numerical metrics to wandb as scalars for number charts
    wandb.log({
        "model_info/total_parameters": int(total_params),
        "model_info/trainable_parameters": int(trainable_params),
        "model_info/model_size_mb": float(model_size_mb),
        "model_info/conv_parameters": int(conv_params),
        "model_info/fc_parameters": int(fc_params),
        "model_info/conv_percentage": float(conv_params/total_params*100),
        "model_info/fc_percentage": float(fc_params/total_params*100),
    }, step=0)
    
    
    return total_params


def log_training_step(loss, episode, step_count):
    """Log training step metrics"""
    mlflow.log_metric("loss", loss, step=episode * 10000 + step_count)
    wandb.log({"loss": loss, "step": episode * 10000 + step_count})


def log_episode_metrics(episode_reward, mean_reward, epsilon, buffer_size, episode_length, episode):
    """Log episode metrics to both MLflow and wandb with organized structure"""
    # Organized wandb logging with namespaces
    wandb_metrics = {
        "training/episode_reward": episode_reward,
        "training/mean_reward_100": mean_reward,
        "training/epsilon": epsilon,
        "training/episode_length": episode_length,
        "training/episode": episode,
        "system/buffer_size": buffer_size,
        "system/buffer_utilization": buffer_size / 100000  # Assuming 100k max capacity
    }

    mlflow.log_metrics({
        "episode_reward": episode_reward,
        "mean_reward_100": mean_reward,
        "epsilon": epsilon,
        "buffer_size": buffer_size,
        "episode_length": episode_length
    }, step=episode)

    wandb.log(wandb_metrics, step=episode)


def log_10_episode_average(avg_reward, episode):
    """Log 10-episode average reward"""
    mlflow.log_metric("avg_reward_10", avg_reward, step=episode)
    wandb.log({"avg_reward_10": avg_reward})


def log_solved_episode(episode):
    """Log when environment is solved"""
    mlflow.log_metric("solved_at_episode", episode)
    wandb.log({"solved_at_episode": episode})


# =============================================================================
# CHECKPOINT AND ARTIFACT FUNCTIONS
# =============================================================================

def save_checkpoint(artifacts_dir, episode, model, optimizer, episode_rewards, epsilon, run_id, filename_prefix="pong_dqn_cnn"):
    """Save model checkpoint and log to tracking systems"""
    artifacts_path = Path(artifacts_dir)
    checkpoint_path = artifacts_path / f'{filename_prefix}_checkpoint_ep{episode}.pth'
    
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards),
        'epsilon': epsilon,
        'episode_rewards': episode_rewards,
        'run_id': run_id
    }, checkpoint_path)
    
    # Log checkpoint to MLflow and wandb
    mlflow.log_artifact(str(checkpoint_path))
    wandb.save(str(checkpoint_path))
    return checkpoint_path


def save_training_plot(artifacts_dir, episode_rewards, episode, mean_reward_bound, title_suffix=""):
    """Save training plot to file and log to tracking systems (overwrites same filename)"""
    if len(episode_rewards) <= 1:
        return
    
    artifacts_path = Path(artifacts_dir)
    
    # Calculate mean rewards for plotting (using 10-episode averages)
    plot_episodes = []
    plot_rewards = []
    for i in range(10, len(episode_rewards) + 1, 10):
        plot_episodes.append(i)
        plot_rewards.append(np.mean(episode_rewards[i-10:i]))
    
    # SAVE RAW DATA TO CSV FILE - for future comparisons
    import pandas as pd
    
    # Save detailed episode data
    detailed_data = pd.DataFrame({
        'episode': list(range(1, len(episode_rewards) + 1)),
        'episode_reward': episode_rewards,
        'mean_reward_100': [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]
    })
    detailed_csv_path = artifacts_path / "detailed_episode_rewards.csv"
    detailed_data.to_csv(detailed_csv_path, index=False)
    
    # Save 10-episode averages for plotting comparisons
    plot_data = pd.DataFrame({
        'episode': plot_episodes,
        'mean_reward_10ep': plot_rewards
    })
    plot_csv_path = artifacts_path / "training_progress_data.csv"
    plot_data.to_csv(plot_csv_path, index=False)
    
    # Also log the CSV files to MLflow
    mlflow.log_artifact(str(detailed_csv_path))
    mlflow.log_artifact(str(plot_csv_path))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(plot_episodes, plot_rewards, 'b-', linewidth=2, marker='o')
    plt.title(f'CNN-DQN{title_suffix} Pong Training Progress (Episode {episode})')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward (last 10 episodes)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=mean_reward_bound, color='r', linestyle='--', label=f'Target ({mean_reward_bound})')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(plot_rewards)), plot_rewards, 'g-', linewidth=2)
    plt.title(f'Training Progress Detail{title_suffix}')
    plt.xlabel('Episodes (x10)')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to file - OVERWRITES same filename each time
    plot_path = artifacts_path / "current_training_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(str(plot_path))
    wandb.log({"training_plot": wandb.Image(str(plot_path))})
    plt.close()  # Close to save memory
    return plot_path


def display_training_plot(episode_rewards, mean_reward_bound, title_suffix=""):
    """Display training plot to screen"""
    if len(episode_rewards) <= 1:
        return
    
    from IPython.display import clear_output
    
    # Calculate mean rewards for plotting (using 10-episode averages)
    plot_episodes = []
    plot_rewards = []
    for i in range(10, len(episode_rewards) + 1, 10):
        plot_episodes.append(i)
        plot_rewards.append(np.mean(episode_rewards[i-10:i]))
    
    clear_output(wait=True)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(plot_episodes, plot_rewards, 'b-', linewidth=2, marker='o')
    plt.title(f'CNN-DQN{title_suffix} Pong Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward (last 10 episodes)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=mean_reward_bound, color='r', linestyle='--', label=f'Target ({mean_reward_bound})')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(plot_rewards)), plot_rewards, 'g-', linewidth=2)
    plt.title(f'Training Progress Detail{title_suffix}')
    plt.xlabel('Episodes (x10)')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def log_final_results(model, episode, mean_reward, episode_rewards, artifacts_dir, run_id, filename_prefix="final_pong_dqn_cnn"):
    """Log final model and metrics"""
    artifacts_path = Path(artifacts_dir)
    
    # Log final model
    mlflow.pytorch.log_model(model, "final_model")
    
    # Log final metrics
    final_metrics = {
        "final_episode": episode,
        "final_mean_reward": mean_reward,
        "total_episodes": len(episode_rewards)
    }
    
    mlflow.log_metrics(final_metrics)
    wandb.log(final_metrics)
    
    # Save final model to wandb
    final_model_path = artifacts_path / f'{filename_prefix}_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'run_id': run_id,
        'final_metrics': final_metrics
    }, final_model_path)
    wandb.save(str(final_model_path))
    
    return final_model_path


# =============================================================================
# WEIGHT VISUALIZATION FUNCTIONS
# =============================================================================













def log_layer_wise_gradient_norms(model, episode):
    """Log gradient norms for each layer"""
    grad_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms[f"grad_norms/{name}"] = grad_norm
    
    wandb.log(grad_norms, step=episode)








# =============================================================================
# COMPREHENSIVE TRAINING LOGGING INTEGRATION
# =============================================================================

def setup_weight_tracking(model, log_freq=1000, log_all=True):
    """Setup wandb watch for automatic weight tracking"""
    wandb.watch(model, log_freq=log_freq, log="all" if log_all else "gradients")


# =============================================================================
# ENHANCED WEIGHT AND BIAS ANALYSIS FUNCTIONS
# =============================================================================









def log_weight_stats(model, episode):
    """Log weight statistics to wandb (legacy function for compatibility)"""
    weight_stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.data.cpu().numpy()
            weight_stats.update({
                f"weight_stats/{name}_mean": np.mean(weights),
                f"weight_stats/{name}_std": np.std(weights),
                f"weight_stats/{name}_min": np.min(weights),
                f"weight_stats/{name}_max": np.max(weights),
                f"weight_stats/{name}_norm": np.linalg.norm(weights),
                f"weight_stats/{name}_sparsity": np.mean(np.abs(weights) < 1e-6),
            })
    
    wandb.log(weight_stats, step=episode)






