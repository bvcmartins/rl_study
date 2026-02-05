# LOGGING AND VISUALIZATION UTILITIES FOR SPACE INVADERS PPO CNN
# Adapted from pong_ppo_cnn logging utilities

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

def log_hyperparameters(hyperparams):
    """Log hyperparameters to MLflow."""
    filtered_params = {}
    for key, value in hyperparams.items():
        if isinstance(value, (int, float, bool, str)):
            filtered_params[key] = value

    mlflow.log_params(filtered_params)


def log_model_info(model):
    """Log model information to MLflow and wandb."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    conv_params = sum(p.numel() for name, p in model.named_parameters() if 'conv' in name)
    actor_params = sum(p.numel() for name, p in model.named_parameters() if 'actor' in name or 'policy' in name)
    critic_params = sum(p.numel() for name, p in model.named_parameters() if 'critic' in name or 'value' in name)

    model_size_mb = total_params * 4 / (1024 * 1024)

    model_metrics = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "conv_parameters": conv_params,
        "actor_parameters": actor_params,
        "critic_parameters": critic_params,
        "model_size_mb": model_size_mb
    }

    mlflow.log_params(model_metrics)

    wandb.log({
        "model_info/total_parameters": int(total_params),
        "model_info/trainable_parameters": int(trainable_params),
        "model_info/conv_parameters": int(conv_params),
        "model_info/actor_parameters": int(actor_params),
        "model_info/critic_parameters": int(critic_params),
        "model_info/model_size_mb": float(model_size_mb),
        "model_info/conv_percentage": float(conv_params/total_params*100),
        "model_info/actor_percentage": float(actor_params/total_params*100),
        "model_info/critic_percentage": float(critic_params/total_params*100),
    }, step=0)

    return total_params


def log_episode_metrics(episode_reward, mean_reward, episode_length, episode, total_updates):
    """Log episode metrics to MLflow and wandb."""
    wandb_metrics = {
        "training/episode_reward": episode_reward,
        "training/mean_reward_100": mean_reward,
        "training/episode_length": episode_length,
        "training/episode": episode,
        "system/total_updates": total_updates
    }

    mlflow.log_metrics({
        "episode_reward": episode_reward,
        "mean_reward_100": mean_reward,
        "episode_length": episode_length,
        "total_updates": total_updates
    }, step=episode)

    wandb.log(wandb_metrics, step=total_updates)


def log_a2c_losses(policy_loss, value_loss, entropy_loss, total_loss, episode, total_updates):
    """Log PPO/A2C loss components to wandb."""
    wandb.log({
        "losses/policy_loss": policy_loss,
        "losses/value_loss": value_loss,
        "losses/entropy_loss": entropy_loss,
        "losses/total_loss": total_loss
    }, step=total_updates)

    mlflow.log_metrics({
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "total_loss": total_loss
    }, step=episode)


# =============================================================================
# CHECKPOINT AND ARTIFACT FUNCTIONS
# =============================================================================

def save_checkpoint(artifacts_dir, episode, model, optimizer, episode_rewards, run_id, total_updates):
    """Save model checkpoint and log to tracking systems."""
    artifacts_path = Path(artifacts_dir)
    checkpoint_path = artifacts_path / f'space_invaders_ppo_cnn_checkpoint_ep{episode}.pth'

    torch.save({
        'episode': episode,
        'total_updates': total_updates,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards),
        'episode_rewards': episode_rewards,
        'run_id': run_id
    }, checkpoint_path)

    mlflow.log_artifact(str(checkpoint_path))
    wandb.save(str(checkpoint_path))
    return checkpoint_path


def save_training_plot(artifacts_dir, episode_rewards, episode, mean_reward_bound):
    """Save training plot to file and log to tracking systems."""
    if len(episode_rewards) <= 1:
        return

    artifacts_path = Path(artifacts_dir)

    import pandas as pd

    detailed_data = pd.DataFrame({
        'episode': list(range(1, len(episode_rewards) + 1)),
        'episode_reward': episode_rewards,
        'mean_reward_100': [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]
    })
    detailed_csv_path = artifacts_path / "detailed_episode_rewards.csv"
    detailed_data.to_csv(detailed_csv_path, index=False)

    if len(episode_rewards) >= 10:
        plot_episodes = list(range(10, len(episode_rewards) + 1, 10))
        plot_rewards = [np.mean(episode_rewards[i-10:i]) for i in plot_episodes]
        plot_data = pd.DataFrame({
            'episode': plot_episodes,
            'mean_reward_10ep': plot_rewards
        })
        plot_csv_path = artifacts_path / "training_progress_data.csv"
        plot_data.to_csv(plot_csv_path, index=False)
        mlflow.log_artifact(str(plot_csv_path))

    mlflow.log_artifact(str(detailed_csv_path))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    if len(episode_rewards) >= 100:
        moving_avg = [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]
        plt.plot(moving_avg, label='Mean (100 episodes)', linewidth=2)
    plt.axhline(y=mean_reward_bound, color='r', linestyle='--', label=f'Target ({mean_reward_bound})')
    plt.title(f'PPO Training Progress (Episode {episode})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if len(episode_rewards) >= 10:
        plot_episodes = list(range(10, len(episode_rewards) + 1, 10))
        plot_rewards = [np.mean(episode_rewards[i-10:i]) for i in plot_episodes]
        plt.plot(plot_episodes, plot_rewards, 'g-', linewidth=2)
        plt.axhline(y=mean_reward_bound, color='r', linestyle='--', label=f'Target ({mean_reward_bound})')
    plt.title('10-Episode Average')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = artifacts_path / "current_training_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(str(plot_path))
    wandb.log({"training_plot": wandb.Image(str(plot_path))})
    plt.close()
    return plot_path


def display_training_plot(episode_rewards, mean_reward_bound):
    """Display training plot to screen."""
    if len(episode_rewards) <= 1:
        return

    from IPython.display import clear_output

    clear_output(wait=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    if len(episode_rewards) >= 100:
        moving_avg = [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]
        plt.plot(moving_avg, label='Mean (100 episodes)', linewidth=2)
    plt.axhline(y=mean_reward_bound, color='r', linestyle='--', label=f'Target ({mean_reward_bound})')
    plt.title('Space Invaders PPO Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if len(episode_rewards) >= 10:
        plot_episodes = list(range(10, len(episode_rewards) + 1, 10))
        plot_rewards = [np.mean(episode_rewards[i-10:i]) for i in plot_episodes]
        plt.plot(plot_episodes, plot_rewards, 'g-', linewidth=2)
        plt.axhline(y=mean_reward_bound, color='r', linestyle='--', label=f'Target ({mean_reward_bound})')
    plt.title('10-Episode Average')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def log_final_results(model, episode, mean_reward, episode_rewards, artifacts_dir, run_id):
    """Log final model and metrics."""
    artifacts_path = Path(artifacts_dir)

    mlflow.pytorch.log_model(model, "final_model")

    final_metrics = {
        "final_episode": episode,
        "final_mean_reward": mean_reward,
        "total_episodes": len(episode_rewards)
    }

    mlflow.log_metrics(final_metrics)
    wandb.log(final_metrics)

    final_model_path = artifacts_path / 'final_space_invaders_ppo_cnn_model.pth'
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

def log_layer_wise_gradient_norms(model, step):
    """Log gradient norms for each layer."""
    grad_norms = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms[f"grad_norms/{name}"] = grad_norm

    wandb.log(grad_norms, step=step)


def setup_weight_tracking(model, log_freq=1000, log_all=True):
    """Setup wandb watch for automatic weight tracking."""
    wandb.watch(model, log_freq=log_freq, log="all" if log_all else "gradients")


def log_weight_stats(model, step):
    """Log weight statistics to wandb."""
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

    wandb.log(weight_stats, step=step)
