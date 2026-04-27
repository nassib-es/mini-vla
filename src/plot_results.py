import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.append('.')

def plot_benchmark():
    # Load results
    results = np.load('models/benchmark_results.npy', 
                      allow_pickle=True).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.patch.set_facecolor('#0A0E14')
    for ax in axes:
        ax.set_facecolor('#0F1520')
        ax.tick_params(colors='#7A9CC0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1E2D45')

    models  = [r['label'] for r in results]
    colors  = ['#00D4FF', '#81C784']

    # ── Plot 1: Success Rate ──────────────────────────────────────
    ax = axes[0]
    rates = [r['success_rate'] for r in results]
    bars  = ax.bar(models, rates, color=colors, width=0.5)
    ax.axhline(0, color='#F06292', linestyle='--',
               linewidth=1.5, label='Random baseline (~0%)')
    ax.set_title('Success Rate (%)', color='#00D4FF', fontsize=12)
    ax.set_ylabel('%', color='#7A9CC0')
    ax.set_ylim(0, 100)
    ax.legend(facecolor='#141C2B', labelcolor='#F06292', fontsize=8)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', color='white', fontsize=11,
                fontweight='bold')

    # ── Plot 2: Average Reward ────────────────────────────────────
    ax = axes[1]
    rewards = [r['avg_reward'] for r in results]
    bars    = ax.bar(models, rewards, color=colors, width=0.5)
    ax.axhline(-103, color='#F06292', linestyle='--',
               linewidth=1.5, label='Random baseline (~-103)')
    ax.axhline(0, color='#7A9CC0', linestyle=':', linewidth=1)
    ax.set_title('Average Reward', color='#00D4FF', fontsize=12)
    ax.set_ylabel('Reward', color='#7A9CC0')
    ax.set_ylim(-110, 10)
    ax.legend(facecolor='#141C2B', labelcolor='#F06292', fontsize=8)
    for bar, reward in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1 if reward > 0 else bar.get_height() - 4,
                f'{reward:.1f}', ha='center', color='white', fontsize=10)

    # ── Plot 3: Episode outcomes per model ────────────────────────
    ax = axes[2]
    x  = np.arange(len(models))
    w  = 0.25

    for i, r in enumerate(results):
        eps     = r['results']
        landed  = sum(1 for e in eps if e['landed'])
        crashed = sum(1 for e in eps if e['outcome'] == 'CRASHED')
        timeout = sum(1 for e in eps if e['outcome'] == 'TIMEOUT')
        total   = len(eps)

        ax.bar(x[i] - w, landed/total*100,  w, color='#81C784', label='Landed'  if i==0 else '')
        ax.bar(x[i],     crashed/total*100, w, color='#F06292', label='Crashed' if i==0 else '')
        ax.bar(x[i] + w, timeout/total*100, w, color='#FFB74D', label='Timeout' if i==0 else '')

    ax.set_title('Outcome Breakdown (%)', color='#00D4FF', fontsize=12)
    ax.set_ylabel('%', color='#7A9CC0')
    ax.set_xticks(x)
    ax.set_xticklabels(models, color='#7A9CC0')
    ax.set_ylim(0, 100)
    ax.legend(facecolor='#141C2B', labelcolor='#E8EEF7', fontsize=8)

    plt.suptitle('Mini VLA — Model Comparison (Llama 3.1 vs Qwen 2.5)',
                 color='#00D4FF', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('docs/benchmark_results.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0E14')
    plt.show()
    print("Saved to docs/benchmark_results.png")


if __name__ == '__main__':
    plot_benchmark()