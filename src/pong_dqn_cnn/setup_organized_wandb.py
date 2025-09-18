"""
Setup script to organize your existing wandb dashboard
Run this to clean up and reorganize your current wandb project
"""

import wandb
from wandb_dashboard_organizer import setup_organized_wandb_tracking

def setup_wandb_with_organization(project_name, config, model=None):
    """Setup wandb with proper organization from the start"""

    # Initialize with better structure
    wandb.init(
        project=project_name,
        name=f"organized_run_{wandb.util.generate_id()}",
        config=config,
        tags=["organized", "v2"],  # Tag for easy filtering
        notes="Organized dashboard structure"
    )

    # Define metric relationships for better dashboard auto-layout
    wandb.define_metric("training/episode")
    wandb.define_metric("training/*", step_metric="training/episode")
    wandb.define_metric("model/*", step_metric="training/episode")
    wandb.define_metric("weights/*", step_metric="training/episode")
    wandb.define_metric("biases/*", step_metric="training/episode")
    wandb.define_metric("gradients/*", step_metric="training/episode")
    wandb.define_metric("performance/*", step_metric="training/episode")
    wandb.define_metric("system/*", step_metric="training/episode")

    # Setup model watching if model provided
    if model:
        wandb.watch(model, log_freq=500, log="all", log_graph=True)

    print("✅ WandB organized tracking setup complete!")
    print("📊 Your dashboard will now have structured sections:")
    print("   • training/* - Training metrics")
    print("   • model/* - Model information")
    print("   • weights/* - Weight analysis")
    print("   • biases/* - Bias analysis")
    print("   • gradients/* - Gradient tracking")
    print("   • performance/* - Performance metrics")
    print("   • system/* - System metrics")

def migrate_existing_run_to_organized():
    """Helper to tag and organize existing runs"""
    api = wandb.Api()

    # You can run this to tag existing runs for better organization
    print("🔄 To organize existing runs, use wandb web interface to:")
    print("   1. Filter runs by date/performance")
    print("   2. Add tags: 'baseline', 'experiment', 'final'")
    print("   3. Archive old/failed runs")
    print("   4. Create custom dashboard views")

if __name__ == "__main__":
    print("Use this module to setup organized wandb tracking in your notebook")