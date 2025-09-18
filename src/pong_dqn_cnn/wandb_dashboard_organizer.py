"""
WandB Dashboard Organization Utilities
Provides structured logging and dashboard cleanup for better organization
"""

import wandb
import numpy as np
from typing import Dict, Any, List

class WandbOrganizer:
    """Organizes wandb logging with structured namespaces and clean dashboards"""

    def __init__(self):
        self.metric_categories = {
            'training': ['loss', 'episode_reward', 'mean_reward_100', 'epsilon', 'buffer_size'],
            'model': ['total_parameters', 'trainable_parameters', 'model_size_mb'],
            'weights': ['weight_norms', 'weight_stats', 'weights'],
            'biases': ['bias_norms', 'bias_stats', 'biases'],
            'gradients': ['grad_norms'],
            'analysis': ['sparsity'],
            'performance': ['episode_length', 'avg_reward_10', 'solved_at_episode']
        }

    def log_organized_metrics(self, metrics: Dict[str, Any], step: int = None, category: str = None):
        """Log metrics with organized structure"""
        if category:
            # Prefix all metrics with category
            organized_metrics = {f"{category}/{key}": value for key, value in metrics.items()}
        else:
            # Auto-categorize based on metric names
            organized_metrics = {}
            for key, value in metrics.items():
                category = self._auto_categorize_metric(key)
                organized_metrics[f"{category}/{key}"] = value

        wandb.log(organized_metrics, step=step)

    def _auto_categorize_metric(self, metric_name: str) -> str:
        """Auto-categorize metrics based on name patterns"""
        for category, keywords in self.metric_categories.items():
            if any(keyword in metric_name.lower() for keyword in keywords):
                return category
        return 'misc'

    def create_dashboard_sections(self):
        """Create organized dashboard sections"""
        dashboard_config = {
            'training_metrics': {
                'charts': ['training/loss', 'training/episode_reward', 'training/epsilon'],
                'layout': 'grid'
            },
            'model_analysis': {
                'charts': ['weights/*', 'biases/*', 'gradients/*'],
                'layout': 'tabs'
            },
            'performance_tracking': {
                'charts': ['performance/avg_reward_10', 'performance/episode_length'],
                'layout': 'single_column'
            }
        }
        return dashboard_config

    def clean_metric_names(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up metric names for better readability"""
        cleaned = {}
        for key, value in metrics.items():
            # Remove redundant prefixes and clean names
            clean_key = key.replace('_', ' ').title()
            clean_key = clean_key.replace('Fc', 'FC').replace('Conv', 'Conv')
            cleaned[clean_key] = value
        return cleaned

# Enhanced logging functions with better organization
def log_training_metrics_organized(episode_reward, mean_reward, epsilon, buffer_size, episode_length, episode):
    """Organized training metrics logging"""
    organizer = WandbOrganizer()

    training_metrics = {
        'episode_reward': episode_reward,
        'mean_reward_100': mean_reward,
        'epsilon': epsilon,
        'episode_length': episode_length,
        'episode': episode
    }

    buffer_metrics = {
        'buffer_size': buffer_size,
        'buffer_utilization': buffer_size / 100000  # Assuming 100k max
    }

    organizer.log_organized_metrics(training_metrics, step=episode, category='training')
    organizer.log_organized_metrics(buffer_metrics, step=episode, category='system')

def log_model_analysis_organized(model, episode):
    """Organized model analysis logging"""
    organizer = WandbOrganizer()

    # Weight analysis
    weight_metrics = {}
    bias_metrics = {}
    layer_metrics = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data.cpu().numpy()
            layer_clean_name = name.replace('.weight', '').replace('.bias', '')

            if 'bias' in name:
                bias_metrics[f'{layer_clean_name}_mean'] = float(np.mean(data))
                bias_metrics[f'{layer_clean_name}_norm'] = float(np.linalg.norm(data))
            else:
                weight_metrics[f'{layer_clean_name}_mean'] = float(np.mean(data))
                weight_metrics[f'{layer_clean_name}_norm'] = float(np.linalg.norm(data))
                weight_metrics[f'{layer_clean_name}_std'] = float(np.std(data))

    organizer.log_organized_metrics(weight_metrics, step=episode, category='weights')
    organizer.log_organized_metrics(bias_metrics, step=episode, category='biases')

def create_custom_dashboard_layout():
    """Create custom dashboard layout configuration"""
    return {
        'sections': [
            {
                'name': 'Training Progress',
                'metrics': ['training/episode_reward', 'training/mean_reward_100', 'training/epsilon'],
                'chart_type': 'line',
                'time_range': 'last_1000'
            },
            {
                'name': 'Model Health',
                'metrics': ['weights/*_norm', 'biases/*_norm'],
                'chart_type': 'multi_line',
                'grouping': 'by_layer'
            },
            {
                'name': 'Performance Summary',
                'metrics': ['performance/avg_reward_10', 'system/buffer_utilization'],
                'chart_type': 'summary_stats'
            }
        ]
    }

def setup_organized_wandb_tracking(model, config):
    """Setup organized wandb tracking from the start"""
    # Configure wandb with better organization
    wandb.config.update({
        'model_architecture': {
            'total_params': sum(p.numel() for p in model.parameters()),
            'layers': [name for name, _ in model.named_parameters()]
        },
        'training_config': config,
        'dashboard_version': 'organized_v1'
    })

    # Setup metric definitions
    wandb.define_metric("training/episode")
    wandb.define_metric("training/*", step_metric="training/episode")
    wandb.define_metric("weights/*", step_metric="training/episode")
    wandb.define_metric("biases/*", step_metric="training/episode")
    wandb.define_metric("performance/*", step_metric="training/episode")

    return WandbOrganizer()