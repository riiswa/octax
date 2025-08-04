import time
import cv2
import numpy as np
from octax.gymnasium_wrapper import make_gymnasium_env

if __name__ == "__main__":
    # Create Gymnasium-compatible environment
    env = make_gymnasium_env("brix", render_mode="rgb_array")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    def policy(observation):
        return env.action_space.sample()

    # Standard Gymnasium API usage
    obs, info = env.reset(seed=42)
    total_reward = 0
    episode_count = 0
    frames = []
    
    start_time = time.time()
    
    for step in range(1000):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Collect frames after 4 episodes
        if episode_count >= 4:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count} ended. Total reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
    
    end_time = time.time()
    print(f"Steps per second: {1000 / (end_time - start_time):.1f}")
    
    # Display frames
    if frames:
        print(f"Displaying {len(frames)} frames (press 'q' to quit)")
        for frame in frames:
            cv2.imshow('Brix Gameplay', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    
    env.close()