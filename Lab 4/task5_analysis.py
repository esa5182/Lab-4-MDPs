#!/usr/bin/env python3
"""
Task 5: DQN Network Width Analysis
Comparing three different network widths (neurons per layer)
"""

import matplotlib.pyplot as plt
import numpy as np

# Task 5 Data: Network Width Comparison (from DqnLabTest8)
# Narrower Network (fewer neurons per layer)
narrow_epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
narrow_rewards = [-50.00, -2093.24, -121.27, -10001.00, -10001.00, -10001.00, -534.86, -10001.00, -132.00, -10001.00, -937.71, -534.86, -132.00, -736.29, 69.52, -736.29, 69.61, 69.52, -937.71, -132.00, -132.00, 69.88, -534.86, -534.86, -333.43, -534.86, 69.52, 69.79, 69.52, 69.43, 69.90, 69.70, 69.99, 69.72, 69.97, 69.79, 70.27, -131.93, 69.52, 69.70, 69.70, 69.79, 69.52, 69.88, 70.08, 69.61, 69.70, 69.61, 69.63, 69.88]

# Standard Network (medium neurons per layer) 
standard_epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
standard_rewards = [-10001.00, -10001.00, -10001.00, -534.86, -736.29, 69.79, -10001.00, -937.71, -736.29, 69.70, -132.00, 69.79, -132.00, -333.43, -333.43, 69.79, 69.61, 69.52, -131.57, 69.79, -333.43, -132.00, 69.61, 69.70, -131.93, 69.43, 69.88, -131.84, 69.43, 69.61, -131.84, 69.61, 69.79, 69.70, 69.61, -131.93, -131.93, 69.88, 69.61, -131.93, 69.70, 69.52, 69.43, 69.70, 69.70, 69.52, 69.79, 69.79, 69.97, 69.79]

# Wider Network (more neurons per layer)
wide_epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
wide_rewards = [-1610.32, -10001.00, -721.32, 69.76, -131.45, -10001.00, -131.64, -131.84, -131.43, 69.70, 69.79, -534.86, 69.70, -534.86, -131.75, -534.86, -333.43, 69.79, -132.00, 69.61, 69.61, 69.43, 69.52, 69.61, 70.06, 48.00, 69.70, 70.06, 69.72, 69.43, 69.52, 69.70, 69.52, 69.19, 69.70, 69.61, 69.70, 69.61, 69.88, 69.70, 69.61, 69.70, 69.88, 69.52, 69.88, 69.01, 69.52, 69.79, 69.43, 69.79]

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
        'stability_index': 1 / (1 + np.std(rewards_array)),
        'final_performance': np.mean(rewards_array[-10:]),  # Last 10 epochs
        'learning_efficiency': len([i for i in range(1, len(rewards)) if rewards[i] > 0 and rewards[i-1] < 0])
    }

def create_network_width_comparison():
    """Create comprehensive network width comparison visualization"""
    plt.figure(figsize=(20, 14))
    
    # Main learning curves comparison
    plt.subplot(2, 4, 1)
    plt.plot(narrow_epochs, narrow_rewards, 'r-', linewidth=2, alpha=0.8, label='Narrow Network', marker='o', markersize=2)
    plt.plot(standard_epochs, standard_rewards, 'g-', linewidth=2, alpha=0.8, label='Standard Network', marker='s', markersize=2)
    plt.plot(wide_epochs, wide_rewards, 'b-', linewidth=2, alpha=0.8, label='Wide Network', marker='^', markersize=2)
    plt.xlabel('Training Epoch')
    plt.ylabel('Average Reward per Episode')
    plt.title('DQN Network Width Comparison\n(Full Training Performance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-1500, 100)
    
    # Early learning phase (0-20 epochs) - critical period
    plt.subplot(2, 4, 2)
    plt.plot(narrow_epochs[:21], narrow_rewards[:21], 'r-', linewidth=3, alpha=0.8, label='Narrow', marker='o', markersize=4)
    plt.plot(standard_epochs[:21], standard_rewards[:21], 'g-', linewidth=3, alpha=0.8, label='Standard', marker='s', markersize=4)
    plt.plot(wide_epochs[:21], wide_rewards[:21], 'b-', linewidth=3, alpha=0.8, label='Wide', marker='^', markersize=4)
    plt.xlabel('Training Epoch')
    plt.ylabel('Average Reward per Episode')
    plt.title('Early Learning Phase (0-20 Epochs)\nCritical Adaptation Period')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-12000, 100)
    
    # Late learning phase (30-49 epochs) - stable performance
    plt.subplot(2, 4, 3)
    plt.plot(narrow_epochs[30:], narrow_rewards[30:], 'r-', linewidth=3, alpha=0.8, label='Narrow', marker='o', markersize=4)
    plt.plot(standard_epochs[30:], standard_rewards[30:], 'g-', linewidth=3, alpha=0.8, label='Standard', marker='s', markersize=4)
    plt.plot(wide_epochs[30:], wide_rewards[30:], 'b-', linewidth=3, alpha=0.8, label='Wide', marker='^', markersize=4)
    plt.xlabel('Training Epoch')
    plt.ylabel('Average Reward per Episode')
    plt.title('Late Learning Phase (30-49 Epochs)\nSteady-State Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(60, 75)
    
    # Learning stability analysis
    plt.subplot(2, 4, 4)
    
    def calculate_rolling_stability(rewards, window=5):
        """Calculate rolling standard deviation as stability metric"""
        stabilities = []
        for i in range(window-1, len(rewards)):
            window_data = rewards[i-window+1:i+1]
            stability = 1 / (1 + np.std(window_data))  # Higher = more stable
            stabilities.append(stability)
        return stabilities
    
    narrow_stability = calculate_rolling_stability(narrow_rewards)
    standard_stability = calculate_rolling_stability(standard_rewards)
    wide_stability = calculate_rolling_stability(wide_rewards)
    
    stability_epochs = range(4, 50)  # Start from epoch 4 (5-epoch window)
    
    plt.plot(stability_epochs, narrow_stability, 'r-', linewidth=2, alpha=0.8, label='Narrow Stability')
    plt.plot(stability_epochs, standard_stability, 'g-', linewidth=2, alpha=0.8, label='Standard Stability')
    plt.plot(stability_epochs, wide_stability, 'b-', linewidth=2, alpha=0.8, label='Wide Stability')
    plt.xlabel('Training Epoch')
    plt.ylabel('Stability Index (Higher = More Stable)')
    plt.title('Learning Stability Over Time\n(5-Epoch Rolling Window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance distribution analysis
    plt.subplot(2, 4, 5)
    
    # Filter for positive rewards only for distribution analysis
    narrow_positive = [r for r in narrow_rewards if r > 0]
    standard_positive = [r for r in standard_rewards if r > 0]
    wide_positive = [r for r in wide_rewards if r > 0]
    
    plt.hist(narrow_positive, bins=15, alpha=0.6, color='red', label=f'Narrow (n={len(narrow_positive)})', density=True)
    plt.hist(standard_positive, bins=15, alpha=0.6, color='green', label=f'Standard (n={len(standard_positive)})', density=True)
    plt.hist(wide_positive, bins=15, alpha=0.6, color='blue', label=f'Wide (n={len(wide_positive)})', density=True)
    plt.xlabel('Positive Reward Values')
    plt.ylabel('Density')
    plt.title('Success Performance Distribution\n(Positive Rewards Only)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Recovery patterns analysis
    plt.subplot(2, 4, 6)
    
    def analyze_recovery_patterns(rewards):
        """Analyze how quickly networks recover from failures"""
        recovery_times = []
        in_failure = False
        failure_start = 0
        
        for i, reward in enumerate(rewards):
            if reward < -500 and not in_failure:  # Start of failure
                in_failure = True
                failure_start = i
            elif reward > 0 and in_failure:  # Recovery
                recovery_time = i - failure_start
                recovery_times.append(recovery_time)
                in_failure = False
        
        return recovery_times
    
    narrow_recoveries = analyze_recovery_patterns(narrow_rewards)
    standard_recoveries = analyze_recovery_patterns(standard_rewards)
    wide_recoveries = analyze_recovery_patterns(wide_rewards)
    
    recovery_data = [narrow_recoveries, standard_recoveries, wide_recoveries]
    labels = ['Narrow', 'Standard', 'Wide']
    colors = ['red', 'green', 'blue']
    
    # Box plot of recovery times
    positions = [1, 2, 3]
    box_data = []
    for recoveries in recovery_data:
        if recoveries:
            box_data.append(recoveries)
        else:
            box_data.append([0])  # No recoveries
    
    bp = plt.boxplot(box_data, positions=positions, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.xlabel('Network Architecture')
    plt.ylabel('Recovery Time (Epochs)')
    plt.title('Failure Recovery Analysis\n(Time to Recover from Failures)')
    plt.grid(True, alpha=0.3)
    
    # Comprehensive statistical comparison
    plt.subplot(2, 4, 7)
    stats_narrow = calculate_statistics(narrow_rewards)
    stats_standard = calculate_statistics(standard_rewards)
    stats_wide = calculate_statistics(wide_rewards)
    
    categories = ['Success\nRate (%)', 'Final\nPerformance', 'Learning\nEfficiency', 'Stability\nIndex×100']
    narrow_values = [stats_narrow['success_rate'], stats_narrow['final_performance'], 
                     stats_narrow['learning_efficiency'], stats_narrow['stability_index']*100]
    standard_values = [stats_standard['success_rate'], stats_standard['final_performance'],
                       stats_standard['learning_efficiency'], stats_standard['stability_index']*100]
    wide_values = [stats_wide['success_rate'], stats_wide['final_performance'],
                   stats_wide['learning_efficiency'], stats_wide['stability_index']*100]
    
    x = np.arange(len(categories))
    width = 0.25
    
    plt.bar(x - width, narrow_values, width, label='Narrow', color='red', alpha=0.7)
    plt.bar(x, standard_values, width, label='Standard', color='green', alpha=0.7)
    plt.bar(x + width, wide_values, width, label='Wide', color='blue', alpha=0.7)
    
    plt.xlabel('Performance Metrics')
    plt.ylabel('Values')
    plt.title('Network Width Statistical Comparison\n(Key Performance Indicators)')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Learning progression with trend analysis
    plt.subplot(2, 4, 8)
    
    def calculate_trend_line(epochs, rewards):
        """Calculate trend line for learning progression"""
        # Filter out extreme negative values for trend calculation
        filtered_data = [(e, r) for e, r in zip(epochs, rewards) if r > -2000]
        if len(filtered_data) < 10:
            return epochs, [0] * len(epochs)
        
        filtered_epochs, filtered_rewards = zip(*filtered_data)
        coeffs = np.polyfit(filtered_epochs, filtered_rewards, 1)
        trend_line = np.polyval(coeffs, epochs)
        return epochs, trend_line
    
    # Calculate and plot trend lines
    narrow_trend_x, narrow_trend_y = calculate_trend_line(narrow_epochs, narrow_rewards)
    standard_trend_x, standard_trend_y = calculate_trend_line(standard_epochs, standard_rewards)
    wide_trend_x, wide_trend_y = calculate_trend_line(wide_epochs, wide_rewards)
    
    plt.plot(narrow_trend_x, narrow_trend_y, 'r--', linewidth=3, alpha=0.8, label='Narrow Trend')
    plt.plot(standard_trend_x, standard_trend_y, 'g--', linewidth=3, alpha=0.8, label='Standard Trend')
    plt.plot(wide_trend_x, wide_trend_y, 'b--', linewidth=3, alpha=0.8, label='Wide Trend')
    
    # Add actual data with lower alpha
    plt.plot(narrow_epochs, narrow_rewards, 'r-', linewidth=1, alpha=0.3)
    plt.plot(standard_epochs, standard_rewards, 'g-', linewidth=1, alpha=0.3)
    plt.plot(wide_epochs, wide_rewards, 'b-', linewidth=1, alpha=0.3)
    
    plt.xlabel('Training Epoch')
    plt.ylabel('Reward Trend')
    plt.title('Learning Progression Trends\n(Linear Trend Analysis)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-200, 100)
    
    plt.tight_layout()
    plt.savefig('task5_network_width_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_narrow, stats_standard, stats_wide

def analyze_network_width_patterns():
    """Analyze specific patterns in network width behavior"""
    print("🔢 TASK 5: Deep Network Width Analysis")
    print("="*60)
    
    # Calculate comprehensive statistics
    stats_narrow, stats_standard, stats_wide = create_network_width_comparison()
    
    print("\n📊 STATISTICAL SUMMARY:")
    print("-" * 40)
    print(f"Narrow Network (Fewer Neurons):")
    print(f"  Average Reward: {stats_narrow['mean']:.2f} (±{stats_narrow['std']:.2f})")
    print(f"  Success Rate: {stats_narrow['success_rate']:.1f}%")
    print(f"  Catastrophic Failures: {stats_narrow['catastrophic_failures']}")
    print(f"  Final Performance: {stats_narrow['final_performance']:.2f}")
    print(f"  Learning Efficiency: {stats_narrow['learning_efficiency']} transitions")
    
    print(f"\nStandard Network (Medium Neurons):")
    print(f"  Average Reward: {stats_standard['mean']:.2f} (±{stats_standard['std']:.2f})")
    print(f"  Success Rate: {stats_standard['success_rate']:.1f}%")
    print(f"  Catastrophic Failures: {stats_standard['catastrophic_failures']}")
    print(f"  Final Performance: {stats_standard['final_performance']:.2f}")
    print(f"  Learning Efficiency: {stats_standard['learning_efficiency']} transitions")
    
    print(f"\nWide Network (More Neurons):")
    print(f"  Average Reward: {stats_wide['mean']:.2f} (±{stats_wide['std']:.2f})")
    print(f"  Success Rate: {stats_wide['success_rate']:.1f}%")
    print(f"  Catastrophic Failures: {stats_wide['catastrophic_failures']}")
    print(f"  Final Performance: {stats_wide['final_performance']:.2f}")
    print(f"  Learning Efficiency: {stats_wide['learning_efficiency']} transitions")
    
    print("\n🎯 KEY NETWORK WIDTH INSIGHTS:")
    print("-" * 40)
    
    # Determine best performer by final performance
    performers = [
        ("Narrow", stats_narrow['final_performance']),
        ("Standard", stats_standard['final_performance']),
        ("Wide", stats_wide['final_performance'])
    ]
    best_final = max(performers, key=lambda x: x[1])
    print(f"🏆 Best Final Performance: {best_final[0]} ({best_final[1]:.2f} avg)")
    
    # Success rate comparison
    success_performers = [
        ("Narrow", stats_narrow['success_rate']),
        ("Standard", stats_standard['success_rate']),
        ("Wide", stats_wide['success_rate'])
    ]
    best_success = max(success_performers, key=lambda x: x[1])
    print(f"🎯 Highest Success Rate: {best_success[0]} ({best_success[1]:.1f}%)")
    
    print(f"\n🔬 DETAILED WIDTH ANALYSIS:")
    print(f"• Narrow Network: Struggled initially but achieved excellent final stability")
    print(f"• Standard Network: Moderate performance with balanced learning")
    print(f"• Wide Network: Most consistent throughout training with fewer extreme failures")
    
    # Learning pattern insights
    print(f"\n⚡ WIDTH-SPECIFIC PATTERNS:")
    print(f"• Narrow: Slow start, strong finish - took time to utilize limited capacity effectively")
    print(f"• Standard: Balanced approach with moderate volatility")
    print(f"• Wide: Smoother learning curve with better early performance")
    
    # Architectural trade-offs
    print(f"\n🧠 NEURAL WIDTH INSIGHTS:")
    print(f"• Width vs Learning Speed: Wider networks learn more smoothly")
    print(f"• Capacity Trade-off: Narrow networks can achieve great final performance")
    print(f"• Stability Pattern: Width provides more consistent learning trajectories")
    
    # Counter-intuitive findings
    print(f"\n🤔 SURPRISING DISCOVERIES:")
    narrow_late_performance = np.mean(narrow_rewards[40:])
    wide_late_performance = np.mean(wide_rewards[40:])
    
    if narrow_late_performance > wide_late_performance:
        print(f"• Narrow network ultimately outperformed wider network in final epochs!")
        print(f"  Narrow final 10 epochs: {narrow_late_performance:.2f}")
        print(f"  Wide final 10 epochs: {wide_late_performance:.2f}")
        print(f"• This suggests that capacity constraints can force more efficient learning")
    
    return stats_narrow, stats_standard, stats_wide

if __name__ == "__main__":
    print("Starting Task 5: Network Width Analysis...")
    analyze_network_width_patterns()
    print("\n✅ Task 5 analysis complete!")