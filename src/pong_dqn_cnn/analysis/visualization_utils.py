"""
Visualization utilities for Pong DQN CNN analysis notebooks.
Shared functions for gameplay visualization and video management.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import gymnasium as gym
from pathlib import Path
from IPython.display import clear_output, HTML, display
from collections import deque


def preprocess_frame(frame):
    """Preprocess a game frame for the CNN model."""
    gray = np.mean(frame, axis=2).astype(np.uint8)
    cropped = gray[34:194, :]
    resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    """Stack multiple frames together for temporal information."""
    def __init__(self, num_frames=2):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame):
        processed_frame = preprocess_frame(frame)
        for _ in range(self.num_frames):
            self.frames.append(processed_frame)
        return self.get_stacked()

    def step(self, frame):
        processed_frame = preprocess_frame(frame)
        self.frames.append(processed_frame)
        return self.get_stacked()

    def get_stacked(self):
        return np.stack(list(self.frames), axis=0)


def visualize_game_frame(raw_frame, processed_frame, stacked_frames, q_values, action,
                        reward, step, total_reward, checkpoint_name, architecture):
    """Show the current game state and agent decision with fixed layout."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'{checkpoint_name} ({architecture}) - Step {step} - Action: {action}, Step Reward: {reward:.1f}, Total: {total_reward:.1f}',
                 fontsize=14, fontweight='bold')

    # Raw game frame - FIXED SIZE
    axes[0].imshow(raw_frame)
    axes[0].set_title('Raw Game Frame')
    axes[0].axis('off')
    axes[0].set_aspect('equal')

    # Processed frame that agent sees - FIXED SIZE
    axes[1].imshow(processed_frame, cmap='gray')
    axes[1].set_title('Processed Frame (84x84)')
    axes[1].axis('off')
    axes[1].set_aspect('equal')

    # Q-values with FIXED AXES and POSITION
    actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    colors = ['red' if i == action else 'blue' for i in range(len(q_values))]

    # Create bars with fixed positions
    x_positions = np.arange(len(q_values))
    bars = axes[2].bar(x_positions, q_values, color=colors, alpha=0.7, width=0.8)

    # FIXED AXES LIMITS - prevents movement and resizing
    axes[2].set_ylim(-5, 5)  # Fixed Y range for Q-values
    axes[2].set_xlim(-0.5, len(q_values) - 0.5)  # Fixed X range

    axes[2].set_title('Q-Values (Selected Action in Red)')
    axes[2].set_xlabel('Actions')
    axes[2].set_ylabel('Q-Value')
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels(actions, rotation=45, fontsize=10)
    axes[2].grid(True, alpha=0.3)

    # Add horizontal line at y=0 for reference
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Add value labels with fixed positioning
    for i, (bar, val) in enumerate(zip(bars, q_values)):
        # Position text at fixed height relative to bar
        text_y = max(val + 0.1, -4.8) if val >= 0 else min(val - 0.1, 4.8)
        axes[2].text(bar.get_x() + bar.get_width()/2., text_y,
                    f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top',
                    fontweight='bold' if i == action else 'normal',
                    fontsize=9)

    # Frame difference (movement detection) - FIXED SIZE
    frame_diff = np.abs(stacked_frames[1] - stacked_frames[0])
    axes[3].imshow(frame_diff, cmap='hot', vmin=0, vmax=1)  # Fixed color scale
    axes[3].set_title('Movement Detection')
    axes[3].axis('off')
    axes[3].set_aspect('equal')

    # Set fixed positions for all subplots to prevent movement
    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_position([0.05, 0.1, 0.2, 0.8])  # Raw frame
        elif i == 1:
            ax.set_position([0.27, 0.1, 0.2, 0.8])  # Processed frame
        elif i == 2:
            ax.set_position([0.50, 0.1, 0.25, 0.8])  # Q-values (slightly wider)
        elif i == 3:
            ax.set_position([0.77, 0.1, 0.2, 0.8])  # Movement detection

    # Force the figure to maintain its size
    plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.15)
    plt.show()


def play_and_visualize_game(model, device, checkpoint_name, architecture,
                           visualization_frame_skip=10, save_video=False, video_path=None):
    """Play one complete game and show frames based on visualization_frame_skip parameter.

    Args:
        model: The trained model
        device: torch device
        checkpoint_name: Name of checkpoint for display
        architecture: Architecture name for display
        visualization_frame_skip: Show every Nth step of the game
        save_video: Whether to save video of the gameplay
        video_path: Path to save video (required if save_video=True)

    Returns:
        tuple: (final_reward, game_length)
    """
    import torch

    env = gym.make('PongNoFrameskip-v4')

    print(f"Playing game with {checkpoint_name} using {architecture}")
    print(f"Showing every {visualization_frame_skip} step(s) of the game")
    if save_video:
        print(f"Saving video to: {video_path}")
    print("=" * 60)

    state, _ = env.reset()
    frame_stack = FrameStack(2)
    stacked_state = frame_stack.reset(state)

    total_reward = 0
    step_count = 0
    done = False

    # Video writer setup
    video_writer = None
    if save_video and video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (160, 210))

    while not done:
        # Get action from model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = q_values.max(1)[1].item()
            q_vals_numpy = q_values.cpu().numpy()[0]

        # Execute action with frame skipping (4 frames as used in training)
        step_reward = 0
        for _ in range(4):  # Frame skip during game execution (same as training)
            next_state, reward, terminated, truncated, _ = env.step(action)
            step_reward += reward

            # Save frame to video
            if video_writer is not None:
                frame_bgr = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            if terminated or truncated:
                break

        done = terminated or truncated
        total_reward += step_reward

        # Show visualization based on visualization_frame_skip parameter
        if step_count % visualization_frame_skip == 0:
            processed_current = preprocess_frame(next_state)
            clear_output(wait=True)
            visualize_game_frame(
                raw_frame=next_state,
                processed_frame=processed_current,
                stacked_frames=stacked_state,
                q_values=q_vals_numpy,
                action=action,
                reward=step_reward,
                step=step_count,
                total_reward=total_reward,
                checkpoint_name=checkpoint_name,
                architecture=architecture
            )

            print(f"Step {step_count}: Action={action}, Reward={step_reward:.1f}, Total={total_reward:.1f}")

            # Adaptive delay based on frame skip - more frequent = shorter delay
            delay_time = max(0.1, min(0.5, visualization_frame_skip * 0.02))
            time.sleep(delay_time)

        # Update state
        stacked_state = frame_stack.step(next_state)
        step_count += 1

    env.close()

    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_path}")

    print(f"\nGame finished!")
    print(f"Final reward: {total_reward}")
    print(f"Game length: {step_count} steps")
    print(f"Frames visualized: {(step_count // visualization_frame_skip) + 1}")
    print("=" * 60)

    return total_reward, step_count


def check_existing_videos(checkpoints):
    """Check which checkpoints already have gameplay videos.

    Args:
        checkpoints: List of checkpoint dictionaries with 'path' and 'filename' keys

    Returns:
        dict: Dictionary mapping checkpoint filenames to video info (path, size_mb)
    """
    if not checkpoints:
        return {}

    artifacts_dir = Path(checkpoints[0]['path']).parent
    existing_videos = {}

    for checkpoint in checkpoints:
        video_filename = checkpoint['filename'].replace('.pth', '_gameplay.mp4')
        video_path = artifacts_dir / video_filename

        if video_path.exists():
            # Get video file size
            size_mb = video_path.stat().st_size / (1024 * 1024)
            existing_videos[checkpoint['filename']] = {
                'path': video_path,
                'size_mb': size_mb
            }

    return existing_videos


def display_video_grid(video_paths, titles):
    """Display multiple videos in a grid layout.

    Args:
        video_paths: List of paths to video files
        titles: List of titles for each video
    """
    html = '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'

    for video_path, title in zip(video_paths, titles):
        html += f'''
        <div style="flex: 1; min-width: 300px; max-width: 500px;">
            <h3>{title}</h3>
            <video width="100%" controls>
                <source src="{video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        '''

    html += '</div>'
    display(HTML(html))


def extract_episode_or_step_number(checkpoint):
    """Extract episode or step number from checkpoint filename.

    Args:
        checkpoint: Dictionary with 'filename' key

    Returns:
        int: Episode or step number, or 0 if not found
    """
    filename = checkpoint['filename']
    try:
        # Try episode format first
        if 'ep' in filename:
            ep_start = filename.find('ep') + 2
            ep_end = filename.find('.pth')
            if ep_end > ep_start:
                return int(filename[ep_start:ep_end])

        # Try step format
        if 'step' in filename:
            step_start = filename.find('step') + 4
            step_end = filename.find('.pth')
            if step_end > step_start:
                return int(filename[step_start:step_end])

        # Try episode_ format
        if 'episode_' in filename:
            ep_start = filename.find('episode_') + 8
            ep_end = filename.find('.pth')
            if ep_end > ep_start:
                return int(filename[ep_start:ep_end])

        return 0  # Default for files without recognizable numbers
    except (ValueError, AttributeError):
        return 0  # Default for files that can't be parsed
