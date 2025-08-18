"""Quick test of SBX PPO with Octax."""

from octax.gymnasium_wrapper import make_gymnasium_env
from sbx import PPO
import time

# Create environment
env = make_gymnasium_env("blinky")
print(f"✅ Environment created: {env.action_space}, {env.observation_space}")

# Create model with minimal training
model = PPO("MlpPolicy", env, verbose=1, n_steps=64, batch_size=32)
print("✅ Model created")

# Quick training test
print("🏃 Starting quick training test...")
start_time = time.time()
model.learn(total_timesteps=1000, progress_bar=True)
end_time = time.time()

print(f"✅ Training completed in {end_time - start_time:.1f} seconds")

# Quick evaluation
obs, _ = env.reset()
for _ in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

print("✅ Evaluation completed successfully")
print("🎉 SBX PPO works with Octax!")