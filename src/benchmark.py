import sys
import time
import numpy as np
sys.path.append('.')

from env.drone_env import DroneEnv
from src.vla_agent import VLAAgent


def run_benchmark(platform_speed, llm_every, n_episodes=10, verbose=False):
    """Run benchmark with specific configuration."""
    env   = DroneEnv(platform_speed=platform_speed)
    agent = VLAAgent(verbose=False, llm_every=llm_every)

    results = []

    for ep in range(n_episodes):
        state  = env.reset()
        agent.reset()
        total_reward = 0
        steps  = 0
        done   = False
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

        results.append({
            'landed':  landed,
            'reward':  total_reward,
            'steps':   steps,
            'outcome': outcome
        })

        if verbose:
            print(f"  Ep {ep+1:2d} | {outcome:7s} | "
                  f"Steps: {steps:3d} | Reward: {total_reward:7.1f}")

    success_rate = np.mean([r['landed'] for r in results]) * 100
    avg_reward   = np.mean([r['reward'] for r in results])
    avg_steps    = np.mean([r['steps']  for r in results])

    return {
        'success_rate': success_rate,
        'avg_reward':   avg_reward,
        'avg_steps':    avg_steps,
        'results':      results
    }


def full_benchmark():
    """
    Benchmark comparing Llama3.1 vs Qwen2.5.
    Fixed platform speed, 20 episodes each for statistical reliability.
    """
    print("=" * 70)
    print("Mini VLA — Model Comparison Benchmark")
    print("Platform speed: 0.8 | LLM every 5 steps | 20 episodes each")
    print("=" * 70)

    configs = [
        {"model": "llama3.1", "platform_speed": 0.8, "llm_every": 5,
         "label": "Llama 3.1 8B"},
        {"model": "qwen2.5",  "platform_speed": 0.8, "llm_every": 5,
         "label": "Qwen 2.5 7B"},
    ]

    all_results = []

    # Generate fixed seeds before running any model
    seeds = [np.random.randint(0, 10000) for _ in range(20)]

    for cfg in configs:
        print(f"\nRunning: {cfg['label']} (20 episodes)...")

        env   = DroneEnv(platform_speed=cfg['platform_speed'])
        agent = VLAAgent(model=cfg['model'], verbose=False,
                         llm_every=cfg['llm_every'])

        results = []
        start   = time.time()

        for ep in range(20):
            # Use same seed for same episode across models
            np.random.seed(seeds[ep])
            state  = env.reset()
            agent.reset()
            total_reward = 0
            steps  = 0
            done   = False
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

            results.append({
                'landed': landed, 'reward': total_reward,
                'steps': steps,   'outcome': outcome
            })
            print(f"  Ep {ep+1:2d}/20 | {outcome:7s} | "
                  f"Reward: {total_reward:7.1f}")

        elapsed      = time.time() - start
        success_rate = np.mean([r['landed'] for r in results]) * 100
        avg_reward   = np.mean([r['reward'] for r in results])
        avg_steps    = np.mean([r['steps']  for r in results])

        summary = {
            'label':        cfg['label'],
            'model':        cfg['model'],
            'success_rate': success_rate,
            'avg_reward':   avg_reward,
            'avg_steps':    avg_steps,
            'results':      results,
            'time':         elapsed
        }
        all_results.append(summary)

        print(f"  → Success: {success_rate:.0f}% | "
              f"Avg Reward: {avg_reward:.1f} | "
              f"Time: {elapsed:.0f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Success%':>9} {'AvgReward':>10} {'AvgSteps':>9}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['label']:<20} {r['success_rate']:>8.0f}% "
              f"{r['avg_reward']:>10.1f} {r['avg_steps']:>9.1f}")
    print(f"{'Random baseline':<20} {'~0%':>9} {'~-103':>10} {'~35':>9}")
    print("=" * 70)

    np.save('models/benchmark_results.npy', all_results, allow_pickle=True)
    print("\nSaved to models/benchmark_results.npy")
    return all_results


if __name__ == '__main__':
    full_benchmark()