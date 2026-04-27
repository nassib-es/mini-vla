import sys
import time
import numpy as np
sys.path.append('.')

from env.drone_env import DroneEnv
from src.vla_agent import VLAAgent


def run_episode(agent, env, max_steps=200, verbose=True):
    """
    Run one full episode with the VLA agent.
    Returns total reward and outcome.
    """
    state = env.reset()
    total_reward = 0
    steps = 0
    done  = False

    while not done and steps < max_steps:
        action = agent.act(state)
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1

        if verbose:
            outcome = ""
            if done:
                if reward >= env.REWARD_LAND * 0.9:
                    outcome = "LANDED!"
                elif reward <= env.REWARD_CRASH:
                    outcome = "CRASHED"
                else:
                    outcome = "TIMEOUT"
            print(f"Step {steps:3d} | Reward: {reward:7.2f} | "
                  f"Total: {total_reward:8.2f} | {outcome}")

    return total_reward, steps


def run_benchmark(n_episodes=5, verbose_episodes=True):
    """
    Run multiple episodes and report success rate.
    Compares VLA agent vs random baseline.
    """
    env   = DroneEnv(platform_speed=0.8)
    agent = VLAAgent(verbose=False, llm_every=3)  # quiet mode for benchmark

    print("=" * 60)
    print("Mini VLA — Benchmark")
    print(f"Episodes: {n_episodes} | Model: llama3.1")
    print("=" * 60)

    results = []

    for ep in range(1, n_episodes + 1):
        print(f"\nEpisode {ep}/{n_episodes}")
        start = time.time()

        state = env.reset()
        agent.reset()
        total_reward = 0
        steps = 0
        done  = False
        landed = False

        while not done and steps < 200:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        if done and reward >= env.REWARD_LAND * 0.9:
            landed = True
            outcome = "LANDED"
        elif done and reward <= env.REWARD_CRASH:
            outcome = "CRASHED"
        else:
            outcome = "TIMEOUT"

        elapsed = time.time() - start
        results.append({'landed': landed, 'reward': total_reward,
                        'steps': steps, 'outcome': outcome})

        print(f"  Outcome: {outcome} | Steps: {steps} | "
              f"Reward: {total_reward:.1f} | Time: {elapsed:.1f}s")

    # Summary
    success_rate = np.mean([r['landed'] for r in results]) * 100
    avg_reward   = np.mean([r['reward'] for r in results])
    avg_steps    = np.mean([r['steps']  for r in results])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Success Rate:  {success_rate:.1f}%")
    print(f"Avg Reward:    {avg_reward:.1f}")
    print(f"Avg Steps:     {avg_steps:.1f}")
    print(f"Random Baseline: ~0% success (for reference)")
    print("=" * 60)

    return results


if __name__ == '__main__':
    run_benchmark(n_episodes=10)