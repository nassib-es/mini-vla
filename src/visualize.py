import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.append('.')

from env.drone_env import DroneEnv
from src.vla_agent import VLAAgent


def run_and_record(model="qwen2.5", llm_every=5):
    """
    Run one episode and record full trajectory for visualization.
    """
    env   = DroneEnv(platform_speed=0.8)
    agent = VLAAgent(model=model, verbose=False, llm_every=llm_every)

    state = env.reset()
    agent.reset()

    trajectory = {
        'drone_x':  [],
        'drone_y':  [],
        'platform_x': [],
        'actions':  [],
        'strategies': [],
        'llm_steps': [],  # steps where LLM was called
    }

    step = 0
    done = False

    while not done and step < 200:
        # Record LLM decision if this is an LLM step
        if step % llm_every == 0:
            trajectory['llm_steps'].append(step)
            trajectory['strategies'].append(agent.current_strategy)

        action = agent.act(state)
        next_state, reward, done = env.step(action)

        # Denormalize for visualization
        x  = state[0] * 10.0
        y  = state[1] * 15.0
        px = state[4] * 10.0

        trajectory['drone_x'].append(x)
        trajectory['drone_y'].append(y)
        trajectory['platform_x'].append(px)
        trajectory['actions'].append(action)

        state = next_state
        step += 1

    # Final outcome
    if done and reward >= env.REWARD_LAND * 0.9:
        outcome = "LANDED"
    elif done and reward <= env.REWARD_CRASH:
        outcome = "CRASHED"
    else:
        outcome = "TIMEOUT"

    trajectory['outcome'] = outcome
    trajectory['total_steps'] = step

    return trajectory


def plot_trajectory(trajectory, save_path='docs/trajectory.png'):
    """
    Plot with two panels: trajectory + LLM decision timeline.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0A0E14')
    
    # Main trajectory plot
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_facecolor('#0F1520')
    ax1.tick_params(colors='#7A9CC0')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#1E2D45')

    drone_x  = trajectory['drone_x']
    drone_y  = trajectory['drone_y']
    platform_x = trajectory['platform_x']

    # Platform zone
    ax1.axhline(1.0, color='#81C784', linestyle='--',
                linewidth=2, label='Landing platform', alpha=0.7)
    ax1.fill_between([-12, 12], 0, 1.0, color='#81C784', alpha=0.1)

    # Drone trajectory
    ax1.plot(drone_x, drone_y, color='#00D4FF', linewidth=2.5,
             label='Drone path', zorder=3)

    # Platform movement
    ax1.plot(platform_x, [1.0]*len(platform_x), color='#81C784',
             linewidth=3, alpha=0.5, label='Platform movement')

    # Start/end markers
    ax1.scatter(drone_x[0], drone_y[0], color='#FFB74D', s=250,
                marker='o', edgecolors='white', linewidths=2.5,
                label='Start', zorder=5)

    if trajectory['outcome'] == 'LANDED':
        end_color, end_marker = '#81C784', 'v'
    else:
        end_color, end_marker = '#F06292', 'X'

    ax1.scatter(drone_x[-1], drone_y[-1], color=end_color, s=350,
                marker=end_marker, edgecolors='white', linewidths=2.5,
                label=f"{trajectory['outcome']}", zorder=5)

    # LLM decision points on trajectory
    for i, step in enumerate(trajectory['llm_steps']):
        if step < len(drone_x):
            strategy = trajectory['strategies'][i]
            x, y = drone_x[step], drone_y[step]
            color = '#FFB74D' if strategy == 'ALIGN' else '#00BFFF'
            ax1.scatter(x, y, color=color, s=150, marker='D',
                        edgecolors='white', linewidths=1.5, zorder=4,
                        alpha=0.9)

    ax1.set_ylabel('Height (m)', color='#7A9CC0', fontsize=11)
    ax1.set_title(f'VLA Agent Trajectory — {trajectory["outcome"]} '
                  f'({trajectory["total_steps"]} steps)',
                  color='#00D4FF', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(facecolor='#141C2B', labelcolor='#E8EEF7',
               fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.1, color='#7A9CC0')
    ax1.set_xlim(-12, 12)
    ax1.set_ylim(0, 16)

    # LLM Decision Timeline
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_facecolor('#0F1520')
    ax2.tick_params(colors='#7A9CC0')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#1E2D45')

    steps = list(range(trajectory['total_steps']))
    
    # Draw strategy zones
    current_strategy = 'ALIGN'
    zone_start = 0
    
    for i, llm_step in enumerate(trajectory['llm_steps']):
        new_strategy = trajectory['strategies'][i]
        if i > 0:  # Fill previous zone
            zone_end = llm_step
            color = '#FFB74D' if current_strategy == 'ALIGN' else '#00BFFF'
            ax2.axvspan(zone_start, zone_end, alpha=0.3, color=color)
            
        # Mark LLM decision point
        color = '#FFB74D' if new_strategy == 'ALIGN' else '#00BFFF'
        ax2.axvline(llm_step, color=color, linewidth=3, alpha=0.8)
        ax2.text(llm_step, 0.5, new_strategy, rotation=90,
                 va='bottom', ha='right', color=color,
                 fontsize=11, fontweight='bold')
        
        current_strategy = new_strategy
        zone_start = llm_step
    
    # Fill final zone
    color = '#FFB74D' if current_strategy == 'ALIGN' else '#00BFFF'
    ax2.axvspan(zone_start, trajectory['total_steps'], alpha=0.3, color=color)

    ax2.set_xlabel('Timestep', color='#7A9CC0', fontsize=11)
    ax2.set_ylabel('Strategy', color='#7A9CC0', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, trajectory['total_steps'])
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.1, color='#7A9CC0', axis='x')
    
    # Legend for strategies
    align_patch = mpatches.Patch(color='#FFB74D', alpha=0.5, label='ALIGN')
    land_patch  = mpatches.Patch(color='#00BFFF', alpha=0.5, label='LAND')
    ax2.legend(handles=[align_patch, land_patch], facecolor='#141C2B',
               labelcolor='#E8EEF7', fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0A0E14')
    plt.show()
    print(f"Saved to {save_path}")

def animate_trajectory(trajectory, save_path='docs/trajectory.gif'):
    """
    Create animated GIF showing drone landing with LLM decisions.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#0A0E14')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0F1520')
        ax.tick_params(colors='#7A9CC0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1E2D45')

    drone_x = trajectory['drone_x']
    drone_y = trajectory['drone_y']
    platform_x = trajectory['platform_x']
    
    # Static elements on trajectory plot
    ax1.axhline(1.0, color='#81C784', linestyle='--',
                linewidth=2, alpha=0.7)
    ax1.fill_between([-12, 12], 0, 1.0, color='#81C784', alpha=0.1)
    ax1.set_xlim(-12, 12)
    ax1.set_ylim(0, 16)
    ax1.set_ylabel('Height (m)', color='#7A9CC0', fontsize=11)
    ax1.grid(True, alpha=0.1, color='#7A9CC0')
    
    # Initialize plot elements
    drone_path, = ax1.plot([], [], color='#00D4FF', linewidth=2.5, alpha=0.7)
    drone_marker, = ax1.plot([], [], 'o', color='#00D4FF', markersize=15,
                             markeredgecolor='white', markeredgewidth=2)
    platform_marker, = ax1.plot([], [], 's', color='#81C784', markersize=12,
                                alpha=0.8)
    llm_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                       fontsize=12, fontweight='bold', color='#FFB74D',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='#141C2B',
                                alpha=0.9, edgecolor='#FFB74D', linewidth=2))
    
    # Timeline plot
    ax2.set_xlim(0, len(drone_x))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Timestep', color='#7A9CC0', fontsize=11)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.1, color='#7A9CC0', axis='x')
    
    progress_line = ax2.axvline(0, color='#00D4FF', linewidth=3)
    
    def init():
        drone_path.set_data([], [])
        drone_marker.set_data([], [])
        platform_marker.set_data([], [])
        llm_text.set_text('')
        return drone_path, drone_marker, platform_marker, llm_text, progress_line
    
    def update(frame):
        # Update trajectory
        drone_path.set_data(drone_x[:frame+1], drone_y[:frame+1])
        drone_marker.set_data([drone_x[frame]], [drone_y[frame]])
        platform_marker.set_data([platform_x[frame]], [1.0])
        
        # Update LLM decision text
        if frame in trajectory['llm_steps']:
            idx = trajectory['llm_steps'].index(frame)
            strategy = trajectory['strategies'][idx]
            llm_text.set_text(f'LLM: {strategy}')
            color = '#FFB74D' if strategy == 'ALIGN' else '#00BFFF'
            llm_text.get_bbox_patch().set_edgecolor(color)
            llm_text.set_color(color)
            
            # Mark strategy zone on timeline
            ax2.axvspan(frame, len(drone_x), alpha=0.2, color=color)
        
        # Update progress line
        progress_line.set_xdata([frame, frame])
        
        # Update title
        ax1.set_title(f'Step {frame}/{len(drone_x)-1}',
                     color='#00D4FF', fontsize=13, fontweight='bold')
        
        return drone_path, drone_marker, platform_marker, llm_text, progress_line
    
    anim = FuncAnimation(fig, update, frames=len(drone_x),
                        init_func=init, blit=True, interval=50)
    
    print("Saving animation (this may take a minute)...")
    writer = PillowWriter(fps=20)
    anim.save(save_path, writer=writer, dpi=100)
    plt.close()
    print(f"Animation saved to {save_path}")


if __name__ == '__main__':
    import random
    
    # Generate a successful landing
    print("=== Searching for successful landing ===")
    seeds_tried = []
    for _ in range(100):
        seed = random.randint(0, 10000)
        seeds_tried.append(seed)
        np.random.seed(seed)
        traj = run_and_record(model="qwen2.5", llm_every=5)
        if traj['outcome'] == 'LANDED':
            print(f"\nFound landing at seed {seed}!")
            print(f"Creating visualizations...")
            plot_trajectory(traj, save_path='docs/trajectory_success.png')
            animate_trajectory(traj, save_path='docs/trajectory_success.gif')
            break
    
    # Generate a crash
    print("\n=== Searching for crash ===")
    for _ in range(100):
        seed = random.randint(0, 10000)
        while seed in seeds_tried:  # avoid duplicates
            seed = random.randint(0, 10000)
        seeds_tried.append(seed)
        np.random.seed(seed)
        traj = run_and_record(model="qwen2.5", llm_every=5)
        if traj['outcome'] == 'CRASHED':
            print(f"\nFound crash at seed {seed}!")
            print(f"Creating visualizations...")
            plot_trajectory(traj, save_path='docs/trajectory_crash.png')
            animate_trajectory(traj, save_path='docs/trajectory_crash.gif')
            break
    
    print("\nDone! Generated both success and crash examples.")