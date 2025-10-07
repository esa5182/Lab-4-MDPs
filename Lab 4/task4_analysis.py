#!/usr/bin/env python3
"""
Task 4: DQN Network Depth Analysis
Comparing three different hidden layer depths
"""

import matplotlib.pyplot as plt
import numpy as np

# Task 4 Data: Network Depth Comparison (from DqnLabTest7)
# 2 Hidden Layers
layer2_epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
layer2_rewards = [70.02, 69.43, -132.00, -323.70, 87.77, 86.39, 88.11, 88.79, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00]

# 3 Hidden Layers
layer3_epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
layer3_rewards = [-50.00, -50.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, -10001.00, 74.03, 79.73, 79.73, 79.46, -1017.00, -322.97, -320.99, -1910.31, -1197.04, -3716.40, -4255.66, -2460.36, -4042.04, -50.00, -847.72, -50.00, -50.00, -50.00, -50.00, -1480.26, -2699.56, -50.00, -50.00, -50.00, -50.00, -50.00, -50.00, -50.00, -50.00, -50.00, -50.00, -50.00, -50.00, -50.00, -50.00, -50.00, -1074.33, -261.00, -50.00, -50.00, -50.00, -50.00]

# 4 Hidden Layers
layer4_epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
layer4_rewards = [-50.00, -131.93, 69.79, -10001.00, 69.43, -131.93, 69.61, 69.70, 69.88, 69.79, 69.79, 69.70, -1135.31, 69.43, -131.93, -121.79, 79.33, -1735.09, -566.34, -1326.62, -971.68, -2995.60, -3171.24, -1736.02, -2815.24, -3717.00, -3537.64, -4257.80, -3336.01, -1886.97, -2496.53, -50.00, -1886.75, -50.00, -1887.23, -50.00, -50.00, -50.00, -50.00, -1277.19, -50.00, -57.67, -57.67, -50.00, -50.00, 145.67, -50.00, -50.00, -464.33, -50.00]

def calculate_statistics(rewards):
    """Calculate comprehensive statistics for reward data"""
    rewards_array = np.array(rewards)
    return {
        'mean': np.mean(rewards_array),
        'std': np.std(rewards_array),
        'min': np.min(rewards_array),
        'max': np.max(rewards_array),
        'median': np.median(rewards_array),
        'success_rate': (rewards_array > 0).mean() * 100,
        'catastrophic_failures': (rewards_array < -1000).sum(),
        'stability_index': 1 / (1 + np.std(rewards_array)),  # Higher = more stable
        'recovery_ability': len([i for i in range(1, len(rewards)) if rewards[i] > 0 and rewards[i-1] < -1000])
    }

def create_network_depth_comparison():
    """Create comprehensive network depth comparison visualization"""
    plt.figure(figsize=(18, 12))
    
    # Main learning curves
    plt.subplot(2, 3, 1)
    plt.plot(layer2_epochs, layer2_rewards, 'r-', linewidth=2, alpha=0.8, label='2 Hidden Layers', marker='o', markersize=3)
    plt.plot(layer3_epochs, layer3_rewards, 'g-', linewidth=2, alpha=0.8, label='3 Hidden Layers', marker='s', markersize=3)
    plt.plot(layer4_epochs, layer4_rewards, 'b-', linewidth=2, alpha=0.8, label='4 Hidden Layers', marker='^', markersize=3)
    plt.xlabel('Training Epoch')
    plt.ylabel('Average Reward per Episode')
    plt.title('DQN Network Depth Comparison\n(Full Training Curves)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-2000, 200)  # Focus on readable range
    
    # Early training phase (0-15 epochs)
    plt.subplot(2, 3, 2)
    plt.plot(layer2_epochs[:16], layer2_rewards[:16], 'r-', linewidth=3, alpha=0.8, label='2 Layers', marker='o', markersize=5)
    plt.plot(layer3_epochs[:16], layer3_rewards[:16], 'g-', linewidth=3, alpha=0.8, label='3 Layers', marker='s', markersize=5)
    plt.plot(layer4_epochs[:16], layer4_rewards[:16], 'b-', linewidth=3, alpha=0.8, label='4 Layers', marker='^', markersize=5)
    plt.xlabel('Training Epoch')
    plt.ylabel('Average Reward per Episode')
    plt.title('Early Training Phase (0-15 Epochs)\nCritical Learning Period')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-12000, 100)
    
    # Success periods identification
    plt.subplot(2, 3, 3)
    
    def identify_success_periods(rewards, threshold=0):
        """Identify continuous periods of success"""
        success_periods = []
        current_period = []
        for i, reward in enumerate(rewards):
            if reward > threshold:
                current_period.append(i)
            else:
                if current_period:
                    success_periods.append(current_period)
                    current_period = []
        if current_period:
            success_periods.append(current_period)
        return success_periods
    
    # Plot success periods
    layer2_success = identify_success_periods(layer2_rewards)
    layer3_success = identify_success_periods(layer3_rewards)
    layer4_success = identify_success_periods(layer4_rewards)
    
    colors = ['red', 'green', 'blue']
    labels = ['2 Layers', '3 Layers', '4 Layers']
    success_data = [layer2_success, layer3_success, layer4_success]
    
    y_positions = [0.8, 0.5, 0.2]
    
    for i, (success_periods, color, label) in enumerate(zip(success_data, colors, labels)):
        y_pos = y_positions[i]
        for period in success_periods:
            plt.barh(y_pos, len(period), left=period[0], height=0.1, 
                    color=color, alpha=0.7, label=f'{label} Success' if period == success_periods[0] else "")
    
    plt.xlabel('Training Epoch')
    plt.ylabel('Network Architecture')
    plt.title('Success Periods Analysis\n(Continuous Positive Performance)')
    plt.yticks(y_positions, labels)
    plt.xlim(0, 50)
    plt.grid(True, alpha=0.3, axis='x')
    plt.legend()
    
    # Performance distribution
    plt.subplot(2, 3, 4)
    
    # Filter out extreme values for better visualization
    layer2_filtered = [r for r in layer2_rewards if r > -5000]
    layer3_filtered = [r for r in layer3_rewards if r > -5000]
    layer4_filtered = [r for r in layer4_rewards if r > -5000]
    
    plt.hist(layer2_filtered, bins=20, alpha=0.6, color='red', label='2 Layers', density=True)
    plt.hist(layer3_filtered, bins=20, alpha=0.6, color='green', label='3 Layers', density=True)
    plt.hist(layer4_filtered, bins=20, alpha=0.6, color='blue', label='4 Layers', density=True)
    plt.xlabel('Reward Value')
    plt.ylabel('Density')
    plt.title('Performance Distribution\n(Excluding Extreme Failures)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistical comparison
    plt.subplot(2, 3, 5)
    stats_2layer = calculate_statistics(layer2_rewards)
    stats_3layer = calculate_statistics(layer3_rewards)
    stats_4layer = calculate_statistics(layer4_rewards)
    
    categories = ['Success\nRate (%)', 'Catastrophic\nFailures', 'Recovery\nAbility', 'Max\nReward']
    layer2_values = [stats_2layer['success_rate'], stats_2layer['catastrophic_failures'], 
                     stats_2layer['recovery_ability'], stats_2layer['max']]
    layer3_values = [stats_3layer['success_rate'], stats_3layer['catastrophic_failures'],
                     stats_3layer['recovery_ability'], max(0, stats_3layer['max'])]  # Handle negative max
    layer4_values = [stats_4layer['success_rate'], stats_4layer['catastrophic_failures'],
                     stats_4layer['recovery_ability'], max(0, stats_4layer['max'])]
    
    x = np.arange(len(categories))
    width = 0.25
    
    plt.bar(x - width, layer2_values, width, label='2 Layers', color='red', alpha=0.7)
    plt.bar(x, layer3_values, width, label='3 Layers', color='green', alpha=0.7)
    plt.bar(x + width, layer4_values, width, label='4 Layers', color='blue', alpha=0.7)
    
    plt.xlabel('Performance Metrics')
    plt.ylabel('Values')
    plt.title('Network Depth Statistical Comparison\n(Key Performance Indicators)')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Learning progression analysis
    plt.subplot(2, 3, 6)
    
    def calculate_moving_average(data, window=5):
        """Calculate moving average to smooth learning curves"""
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Calculate moving averages (excluding extreme negative values for readability)
    layer2_ma = calculate_moving_average([max(r, -2000) for r in layer2_rewards])
    layer3_ma = calculate_moving_average([max(r, -2000) for r in layer3_rewards])  
    layer4_ma = calculate_moving_average([max(r, -2000) for r in layer4_rewards])
    
    epochs_ma = range(len(layer2_ma))
    
    plt.plot(epochs_ma, layer2_ma, 'r-', linewidth=3, alpha=0.8, label='2 Layers (MA)')
    plt.plot(epochs_ma, layer3_ma, 'g-', linewidth=3, alpha=0.8, label='3 Layers (MA)')
    plt.plot(epochs_ma, layer4_ma, 'b-', linewidth=3, alpha=0.8, label='4 Layers (MA)')
    plt.xlabel('Training Epoch')
    plt.ylabel('Moving Average Reward (5-epoch window)')
    plt.title('Learning Progression Trends\n(Smoothed Performance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-2000, 100)
    
    plt.tight_layout()
    plt.savefig('task4_network_depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_2layer, stats_3layer, stats_4layer

def analyze_network_depth_patterns():
    """Analyze specific patterns in network depth behavior"""
    print("🏗️ TASK 4: Deep Network Depth Analysis")
    print("="*60)
    
    # Calculate comprehensive statistics
    stats_2layer, stats_3layer, stats_4layer = create_network_depth_comparison()
    
    print("\n📊 STATISTICAL SUMMARY:")
    print("-" * 40)
    print(f"2 Hidden Layers:")
    print(f"  Average Reward: {stats_2layer['mean']:.2f} (±{stats_2layer['std']:.2f})")
    print(f"  Success Rate: {stats_2layer['success_rate']:.1f}%")
    print(f"  Catastrophic Failures: {stats_2layer['catastrophic_failures']}")
    print(f"  Recovery Ability: {stats_2layer['recovery_ability']} instances")
    print(f"  Best Performance: {stats_2layer['max']:.2f}")
    
    print(f"\n3 Hidden Layers:")
    print(f"  Average Reward: {stats_3layer['mean']:.2f} (±{stats_3layer['std']:.2f})")
    print(f"  Success Rate: {stats_3layer['success_rate']:.1f}%")
    print(f"  Catastrophic Failures: {stats_3layer['catastrophic_failures']}")
    print(f"  Recovery Ability: {stats_3layer['recovery_ability']} instances")
    print(f"  Best Performance: {stats_3layer['max']:.2f}")
    
    print(f"\n4 Hidden Layers:")
    print(f"  Average Reward: {stats_4layer['mean']:.2f} (±{stats_4layer['std']:.2f})")
    print(f"  Success Rate: {stats_4layer['success_rate']:.1f}%")
    print(f"  Catastrophic Failures: {stats_4layer['catastrophic_failures']}")
    print(f"  Recovery Ability: {stats_4layer['recovery_ability']} instances")
    print(f"  Best Performance: {stats_4layer['max']:.2f}")
    
    print("\n🎯 KEY NETWORK DEPTH INSIGHTS:")
    print("-" * 40)
    
    # Determine best performer by success rate
    performers = [
        ("2 Layers", stats_2layer['success_rate']),
        ("3 Layers", stats_3layer['success_rate']),
        ("4 Layers", stats_4layer['success_rate'])
    ]
    best_performer = max(performers, key=lambda x: x[1])
    print(f"🏆 Highest Success Rate: {best_performer[0]} ({best_performer[1]:.1f}%)")
    
    # Learning pattern analysis
    print(f"\n🔬 DETAILED DEPTH ANALYSIS:")
    print(f"• 2 Layers: Started strong but suffered complete collapse after epoch 8")
    print(f"• 3 Layers: Very unstable with alternating success/failure periods")
    print(f"• 4 Layers: Most balanced with sustained good performance periods")
    
    # Early vs late performance
    early_2 = np.mean([r for r in layer2_rewards[:8] if r > -1000])
    early_3 = np.mean([r for r in layer3_rewards[8:12] if r > -1000])  # 3-layer had success later
    early_4 = np.mean([r for r in layer4_rewards[:12] if r > -1000])
    
    print(f"\n⚡ DEPTH-SPECIFIC PATTERNS:")
    print(f"• Shallow (2 layers): Fastest initial learning but catastrophic instability")
    print(f"• Medium (3 layers): Delayed learning with extreme volatility")  
    print(f"• Deep (4 layers): Most sustained learning with recovery capability")
    
    # Architecture insights
    print(f"\n🧠 NEURAL ARCHITECTURE INSIGHTS:")
    print(f"• Depth vs Stability: Deeper networks show better recovery from failures")
    print(f"• Learning Speed: Shallower networks learn faster but less robustly")
    print(f"• Complexity Trade-off: More layers = more stable but harder to train initially")
    
    return stats_2layer, stats_3layer, stats_4layer

if __name__ == "__main__":
    print("Starting Task 4: Network Depth Analysis...")
    analyze_network_depth_patterns()
    print("\n✅ Task 4 analysis complete!")