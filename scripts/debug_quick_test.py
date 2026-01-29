from __future__ import annotations

import json
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from minigrid import register_minigrid_envs
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Register MiniGrid environments
register_minigrid_envs()

# #region agent log
def _log(hypothesis_id, location, message, data):
    try:
        with open('/Users/annafu/gi-claim2test/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': hypothesis_id,
                'location': location,
                'message': message,
                'data': data,
                'timestamp': __import__('time').time() * 1000
            }) + '\n')
    except: pass
# #endregion


def main() -> None:
    """Quick debug test to verify basic training works."""
    env_name = "MiniGrid-DoorKey-5x5-v0"  # Changed to DoorKey which has clearer rewards
    timesteps = 50000
    seed = 0

    print("=" * 60)
    print("DEBUG QUICK TEST")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Timesteps: {timesteps}")
    print(f"Seed: {seed}")
    print()

    def make_env(env_name, seed):
        def _init():
            env = gym.make(env_name)
            env = ImgObsWrapper(env)  # Convert to image observation
            env = FlattenObservation(env)  # Flatten to 1D vector
            env.reset(seed=seed)
            return env
        return _init

    vec_env = DummyVecEnv([make_env(env_name, seed)])

    # Print observation space
    test_env = make_env(env_name, seed)()
    print(f"Observation space: {test_env.observation_space}")
    print(f"Action space: {test_env.action_space}")
    # #region agent log
    _log('A', 'debug_quick_test.py:41', 'Training env observation space', {'obs_space': str(test_env.observation_space), 'action_space': str(test_env.action_space)})
    # #endregion
    
    # Test environment gives rewards and check observation format
    print("\nTesting environment rewards and observations...")
    test_obs, test_info = test_env.reset()
    print(f"  Observation shape: {test_obs.shape if hasattr(test_obs, 'shape') else 'N/A'}")
    print(f"  Observation dtype: {test_obs.dtype if hasattr(test_obs, 'dtype') else 'N/A'}")
    print(f"  Observation range: [{test_obs.min() if hasattr(test_obs, 'min') else 'N/A'}, {test_obs.max() if hasattr(test_obs, 'max') else 'N/A'}]")
    # #region agent log
    _log('A', 'debug_quick_test.py:48', 'Training env observation sample', {'shape': str(test_obs.shape) if hasattr(test_obs, 'shape') else None, 'dtype': str(test_obs.dtype) if hasattr(test_obs, 'dtype') else None, 'min': float(test_obs.min()) if hasattr(test_obs, 'min') else None, 'max': float(test_obs.max()) if hasattr(test_obs, 'max') else None})
    # #endregion
    
    test_reward = 0
    for i in range(20):
        test_action = test_env.action_space.sample()
        test_obs, test_r, test_term, test_trunc, test_info = test_env.step(test_action)
        test_reward += test_r
        # #region agent log
        _log('C', 'debug_quick_test.py:54', 'Environment step reward', {'step': i, 'reward': float(test_r), 'terminated': bool(test_term), 'truncated': bool(test_trunc)})
        # #endregion
        if test_term or test_trunc:
            print(f"  Episode ended at step {i+1} with reward {test_r:.2f}")
            break
    print(f"  Random walk total reward: {test_reward:.2f}")
    # #region agent log
    _log('C', 'debug_quick_test.py:60', 'Environment random walk total', {'total_reward': float(test_reward)})
    # #endregion
    test_env.close()
    print()

    # Train PPO with verbose output
    print("Training PPO agent...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        seed=seed,
        verbose=1,
    )

    # Add callback to track rewards during training
    from stable_baselines3.common.callbacks import BaseCallback
    
    class RewardCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.episode_lengths = []
            
        def _on_step(self) -> bool:
            # Check for episode completion in infos - Stable-Baselines3 puts episode info in infos
            infos = self.locals.get('infos', [])
            for info in infos:
                if isinstance(info, dict):
                    # Episode info can be directly in info dict or nested
                    if 'episode' in info:
                        ep_info = info['episode']
                        if ep_info and isinstance(ep_info, dict):
                            reward = ep_info.get('r', 0)
                            length = ep_info.get('l', 0)
                            self.episode_rewards.append(reward)
                            self.episode_lengths.append(length)
                            # #region agent log
                            _log('B', 'debug_quick_test.py:127', 'Episode completed during training', {'reward': float(reward), 'length': int(length), 'total_episodes': len(self.episode_rewards)})
                            # #endregion
                            if len(self.episode_rewards) % 10 == 0:
                                recent_rewards = self.episode_rewards[-10:]
                                print(f"  Recent episodes: mean_reward={sum(recent_rewards)/len(recent_rewards):.2f}, max={max(recent_rewards):.2f}")
            return True
    
    callback = RewardCallback()
    # #region agent log
    _log('B', 'debug_quick_test.py:96', 'Starting training', {'timesteps': timesteps, 'obs_space': str(vec_env.observation_space)})
    # #endregion
    model.learn(total_timesteps=timesteps, callback=callback)
    # #region agent log
    _log('B', 'debug_quick_test.py:99', 'Training completed', {'episodes': len(callback.episode_rewards), 'mean_reward': float(sum(callback.episode_rewards)/len(callback.episode_rewards)) if callback.episode_rewards else 0.0})
    # #endregion
    
    if callback.episode_rewards:
        print(f"\nTraining summary: {len(callback.episode_rewards)} episodes")
        print(f"  Mean reward: {sum(callback.episode_rewards)/len(callback.episode_rewards):.2f}")
        print(f"  Max reward: {max(callback.episode_rewards):.2f}")
        print(f"  Final 10 episodes mean: {sum(callback.episode_rewards[-10:])/min(10, len(callback.episode_rewards)):.2f}")

    # Quick evaluation - use SAME wrappers as training
    print("\nEvaluating trained agent...")
    env = gym.make(env_name)
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)
    # #region agent log
    _log('A', 'debug_quick_test.py:108', 'Evaluation env observation space', {'obs_space': str(env.observation_space), 'matches_training': str(env.observation_space) == str(test_env.observation_space)})
    # #endregion

    rewards = []
    for episode in range(10):
        obs, info = env.reset(seed=seed + episode)
        # #region agent log
        _log('A', 'debug_quick_test.py:113', 'Evaluation episode reset', {'episode': episode, 'obs_shape': str(obs.shape) if hasattr(obs, 'shape') else None})
        # #endregion
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done and episode_length < 200:  # Limit episode length
            action, _ = model.predict(obs, deterministic=True)
            # #region agent log
            _log('D', 'debug_quick_test.py:120', 'Model prediction', {'episode': episode, 'step': episode_length, 'action': int(action), 'action_valid': bool(0 <= action < env.action_space.n)})
            # #endregion
            obs, reward, terminated, truncated, info = env.step(action)
            # #region agent log
            _log('C', 'debug_quick_test.py:123', 'Evaluation step reward', {'episode': episode, 'step': episode_length, 'reward': float(reward), 'terminated': bool(terminated), 'truncated': bool(truncated)})
            # #endregion
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        success = info.get("success", episode_reward > 0)
        # #region agent log
        _log('E', 'debug_quick_test.py:131', 'Episode complete', {'episode': episode, 'total_reward': float(episode_reward), 'length': episode_length, 'success': bool(success)})
        # #endregion
        print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}, success={success}")

    env.close()

    mean_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Max reward: {max_reward:.2f}")
    if max_reward > 0:
        print("  ✓ SUCCESS: Agent received non-zero rewards!")
    else:
        print("  ✗ WARNING: All rewards are zero")
    print("=" * 60)


if __name__ == "__main__":
    main()
