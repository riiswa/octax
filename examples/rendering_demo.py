"""
Demonstration of Octax rendering capabilities.

This script shows how to use the rendering system to visualize CHIP-8 games
during training or evaluation.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from octax.environments import create_environment, print_metadata


def basic_rendering_demo():
    """Basic example of rendering a single frame."""
    print("🎮 Basic Rendering Demo")
    print("=" * 50)

    # Create environment with rendering enabled
    env, metadata = create_environment(
        "brix", render_mode="rgb_array", render_scale=8
    )

    print_metadata(metadata)

    # Initialize environment
    rng = jax.random.PRNGKey(42)
    state, obs, info = env.reset(rng)

    # Take a few random steps
    for step in range(10):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.randint(action_rng, (), 0, env.num_actions)
        state, obs, reward, terminated, truncated, info = env.step(state, action)

        if terminated or truncated:
            state, obs, info = env.reset(rng)

    # Render the current state
    rgb_frame = env.render(state)

    if rgb_frame is not None:
        print(f"✅ Rendered frame shape: {rgb_frame.shape}")
        print(
            f"   Resolution: {rgb_frame.shape[1]}x{rgb_frame.shape[0]} (upscaled from 64x32)"
        )

        # Display using matplotlib
        plt.figure(figsize=(10, 5))
        plt.imshow(rgb_frame)
        plt.title(f"CHIP-8 Game: {metadata['title']}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("❌ No frame rendered (render_mode is None)")


def color_scheme_demo():
    """Demonstrate different color schemes."""
    print("\n🎨 Color Scheme Demo")
    print("=" * 50)

    schemes = ["octax", "classic", "amber", "white", "blue", "retro"]

    # Create environment
    env, metadata = create_environment("brix", render_mode="rgb_array")

    # Get a game state
    rng = jax.random.PRNGKey(123)
    state, obs, info = env.reset(rng)

    # Take some steps to get interesting display
    for _ in range(50):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.randint(action_rng, (), 0, env.num_actions)
        state, obs, reward, terminated, truncated, info = env.step(state, action)
        if terminated or truncated:
            break

    # Render with different color schemes
    fig, axes = plt.subplots(1, len(schemes), figsize=(15, 3))
    fig.suptitle("CHIP-8 Color Schemes", fontsize=16)

    for i, scheme in enumerate(schemes):
        # Create environment with specific color scheme
        env_scheme, _ = create_environment(
            "brix",
            render_mode="rgb_array",
            color_scheme=scheme,
            render_scale=4,  # Smaller for grid display
        )

        rgb_frame = env_scheme.render(state)

        axes[i].imshow(rgb_frame)
        axes[i].set_title(scheme.capitalize())
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def batch_rendering_demo():
    """Demonstrate batch rendering of multiple game states."""
    print("\n📺 Batch Rendering Demo")
    print("=" * 50)

    from octax.rendering import batch_render

    # Create multiple environment instances
    env, metadata = create_environment("brix", render_mode="rgb_array")

    # Generate multiple game states
    num_states = 6
    displays = []

    for i in range(num_states):
        rng = jax.random.PRNGKey(i * 100)
        state, obs, info = env.reset(rng)

        # Run for different lengths to get variety
        steps = 20 + i * 10
        for _ in range(steps):
            rng, action_rng = jax.random.split(rng)
            action = jax.random.randint(action_rng, (), 0, env.num_actions)
            state, obs, reward, terminated, truncated, info = env.step(state, action)
            if terminated or truncated:
                break

        displays.append(state.display)

    # Stack displays for batch rendering
    displays_array = jnp.stack(displays)

    # Render all displays in a grid
    grid_image = batch_render(displays_array, scale=3, color_scheme="octax")

    print(f"✅ Batch rendered {num_states} states")
    print(f"   Grid image shape: {grid_image.shape}")

    plt.figure(figsize=(12, 8))
    plt.imshow(grid_image)
    plt.title(f"Batch Rendering: {num_states} Game States")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def training_with_visualization():
    """Example of training with periodic visualization."""
    print("\n🏋️ Training with Visualization")
    print("=" * 50)

    env, metadata = create_environment(
        "tank", render_mode="rgb_array", color_scheme="octax"
    )

    def simple_policy(rng, obs):
        """Random policy for demonstration."""
        return jax.random.randint(rng, (), 0, env.num_actions)

    # Training loop with visualization
    rng = jax.random.PRNGKey(0)
    episode_rewards = []
    visualization_frames = []

    for episode in range(5):
        rng, reset_rng = jax.random.split(rng)
        state, obs, info = env.reset(reset_rng)
        episode_reward = 0

        for step in range(1000):
            rng, action_rng = jax.random.split(rng)
            action = simple_policy(action_rng, obs)
            state, obs, reward, terminated, truncated, info = env.step(state, action)
            episode_reward += reward

            # Capture frame every 20 steps
            if step % 20 == 0:
                frame = env.render(state)
                if frame is not None:
                    visualization_frames.append(frame)

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    # Show training progress
    if visualization_frames:
        print(f"\n📈 Captured {len(visualization_frames)} visualization frames")

        # Display first and last frames
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.imshow(visualization_frames[0])
        ax1.set_title("Training Start")
        ax1.axis("off")

        ax2.imshow(visualization_frames[-1])
        ax2.set_title("Training End")
        ax2.axis("off")

        plt.suptitle(f"Training Progress: {metadata['title']}")
        plt.tight_layout()
        plt.show()

        # Plot rewards
        plt.figure(figsize=(8, 4))
        plt.plot(episode_rewards, "o-")
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("🚀 Octax Rendering Demonstration")
    print("================================")

    try:
        basic_rendering_demo()
        color_scheme_demo()
        batch_rendering_demo()
        training_with_visualization()
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Make sure you have matplotlib installed and ROM files available.")
