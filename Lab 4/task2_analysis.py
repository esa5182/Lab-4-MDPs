import matplotlib.pyplot as plt
import numpy as np

def analyze_task2_greed_experiment():
    """Analyze the results of Task 2: DQNs and Greed"""
    print("🎯 Analyzing Task 2: DQNs and Greed Experiment...")
    
    epochs = list(range(50))
    
    # Extract data from console output for three different greedy probabilities
    # HIGH GREED (0.95 - very little exploration)
    high_greed_rewards = [
        -10001.0000, 74.1395, -10001.0000, 58.5971, -10001.0000,
        69.6982, 47.8998, -218.9543, 69.5185, 69.6084,
        -10001.0000, 69.7881, 26.0114, 68.2286, 67.6286,
        69.5185, 69.5185, 65.8286, 69.6084, 47.7200,
        68.2286, 26.0514, 68.8286, 68.2286, 67.6286,
        26.0514, 68.2286, 64.6400, 67.6286, 65.8400,
        66.4286, 69.4286, 67.6286, 67.6286, 67.6286,
        68.2286, 67.6286, 68.2286, 68.2286, 67.0286,
        67.6286, 68.2286, 67.6286, 68.8286, 65.8400,
        67.6286, 67.0400, 67.6286, 61.6629, 65.8400
    ]
    
    # MEDIUM GREED (0.5 - balanced exploration/exploitation)
    med_greed_rewards = [
        -10001.0000, -10001.0000, 69.5185, 80.5426, -2025.9800,
        77.2328, 69.5185, -10001.0000, -10001.0000, 26.0114,
        4.3029, 70.0045, 69.8780, 74.6629, 69.6982,
        69.4286, 69.5185, 69.8963, 69.7881, 69.8780,
        69.6084, 69.5185, 69.5185, 69.6084, 69.7881,
        69.6982, 69.9862, 69.6084, 69.6084, 69.5185,
        69.6982, 69.6084, 69.6982, 69.6084, 69.5185,
        69.6084, 69.5185, 69.9862, 69.7881, 69.6084,
        69.8064, 69.9862, 69.5185, 69.5185, 69.6267,
        69.6084, 69.6084, 69.5185, 69.7881, 69.6084
    ]
    
    # LOW GREED (0.1 - lots of exploration) 
    low_greed_rewards = [
        -10001.0000, -10001.0000, -10001.0000, -10001.0000, -10001.0000,
        85.5200, -10001.0000, 70.0229, 69.8248, 71.9657,
        68.8286, 68.2286, 69.6084, 70.1477, 69.6982,
        68.2286, 64.6400, 67.6286, 68.2286, 69.6982,
        69.5185, 69.5185, 69.6982, 69.8780, 69.7881,
        65.2400, 69.6982, 69.6084, 67.0286, 69.7881,
        69.6982, 69.7881, 62.8514, 69.5185, 65.8286,
        64.6400, 67.6286, 65.8400, 68.8286, 69.8780,
        69.6982, 69.4286, 69.0084, 67.0286, 69.6982,
        69.6084, 69.8780, 68.2286, 65.2400, 65.2400
    ]
    
    # Create comprehensive learning curve comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top plot: Raw learning curves
    ax1.plot(epochs, high_greed_rewards, 'r-', linewidth=2, alpha=0.7, label='High Greed (ε≈0.05)')
    ax1.plot(epochs, med_greed_rewards, 'g-', linewidth=2, alpha=0.7, label='Medium Greed (ε≈0.5)')
    ax1.plot(epochs, low_greed_rewards, 'b-', linewidth=2, alpha=0.7, label='Low Greed (ε≈0.9)')
    
    ax1.axhline(y=70, color='k', linestyle='--', alpha=0.5, label='Target Performance')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax1.set_title('Task 2: DQN Performance with Different Exploration Strategies', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Average Reward per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-2000, 100)  # Zoom in to see normal performance
    
    # Bottom plot: Smoothed curves (moving average) to see trends better
    window_size = 5
    
    def moving_average(data, window):
        # Replace catastrophic failures with NaN for smoothing
        clean_data = [x if x > -1000 else np.nan for x in data]
        result = []
        for i in range(len(clean_data)):
            start = max(0, i - window + 1)
            window_data = [x for x in clean_data[start:i+1] if not np.isnan(x)]
            if window_data:
                result.append(np.mean(window_data))
            else:
                result.append(np.nan)
        return result
    
    high_smooth = moving_average(high_greed_rewards, window_size)
    med_smooth = moving_average(med_greed_rewards, window_size)
    low_smooth = moving_average(low_greed_rewards, window_size)
    
    ax2.plot(epochs, high_smooth, 'r-', linewidth=3, label='High Greed (Smoothed)')
    ax2.plot(epochs, med_smooth, 'g-', linewidth=3, label='Medium Greed (Smoothed)')
    ax2.plot(epochs, low_smooth, 'b-', linewidth=3, label='Low Greed (Smoothed)')
    
    ax2.axhline(y=70, color='k', linestyle='--', alpha=0.5, label='Target Performance')
    ax2.set_title('Smoothed Learning Curves (Excluding Catastrophic Failures)', fontsize=12)
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Average Reward (5-epoch moving average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Task2_Greed_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate statistics
    def analyze_performance(rewards, name):
        normal_rewards = [r for r in rewards if r > -1000]
        catastrophic_failures = len([r for r in rewards if r < -1000])
        
        print(f"\n📊 {name} Analysis:")
        print(f"   Normal performance range: {min(normal_rewards):.2f} to {max(normal_rewards):.2f}")
        print(f"   Average normal performance: {np.mean(normal_rewards):.2f}")
        print(f"   Standard deviation: {np.std(normal_rewards):.2f}")
        print(f"   Catastrophic failures: {catastrophic_failures}/50 epochs ({catastrophic_failures/50*100:.1f}%)")
        print(f"   Final 10 epochs average: {np.mean(normal_rewards[-10:] if len(normal_rewards) >= 10 else normal_rewards):.2f}")
        
        return {
            'normal_rewards': normal_rewards,
            'avg_performance': np.mean(normal_rewards),
            'std_performance': np.std(normal_rewards),
            'catastrophic_failures': catastrophic_failures,
            'stability': np.std(normal_rewards[-10:] if len(normal_rewards) >= 10 else normal_rewards)
        }
    
    high_stats = analyze_performance(high_greed_rewards, "HIGH GREED (Low Exploration)")
    med_stats = analyze_performance(med_greed_rewards, "MEDIUM GREED (Balanced)")
    low_stats = analyze_performance(low_greed_rewards, "LOW GREED (High Exploration)")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   Best average performance: {'Medium Greed' if med_stats['avg_performance'] > max(high_stats['avg_performance'], low_stats['avg_performance']) else 'High Greed' if high_stats['avg_performance'] > low_stats['avg_performance'] else 'Low Greed'}")
    print(f"   Most stable learning: {'High Greed' if high_stats['stability'] < min(med_stats['stability'], low_stats['stability']) else 'Medium Greed' if med_stats['stability'] < low_stats['stability'] else 'Low Greed'}")
    print(f"   Fewest failures: {'High Greed' if high_stats['catastrophic_failures'] < min(med_stats['catastrophic_failures'], low_stats['catastrophic_failures']) else 'Medium Greed' if med_stats['catastrophic_failures'] < low_stats['catastrophic_failures'] else 'Low Greed'}")
    
    return high_stats, med_stats, low_stats

# Run the analysis
high_stats, med_stats, low_stats = analyze_task2_greed_experiment()
print("✅ Task 2 analysis complete!")