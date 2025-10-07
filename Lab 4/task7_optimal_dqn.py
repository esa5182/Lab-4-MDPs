#!/usr/bin/env python3
"""
Task 7: Optimal DQN Design - Experimenting with different configurations
Based on insights from Tasks 1-6
"""

from testsDqnLab import *
import numpy as np

def run_baseline_config():
    """Run the default configuration provided in DqnLabTest10"""
    print("="*60)
    print("EXPERIMENT 1: Baseline Configuration (Default Settings)")
    print("="*60)
    
    seedValue = 42
    seed(seedValue)
    
    numStates = 45
    numActions = 15
    start = 0
    mdp = generate_random_mdp(numStates, numActions)
    
    # Default configuration from test
    probGreedy = .7
    discountFactor = .9
    learningRate = .001
    layer_sizes = [128]
    regularization = .0001
    
    agent = DeepQNetworkAgent("Baseline_DQN", numActions, mdp.numStates, probGreedy, discountFactor, 
                              learningRate, layer_sizes, regularization, seedValue, Verbosity.SILENT)
    
    numSamples = 50
    numEpochs = 50
    maxTrajectoryLen = 100
    
    print(f"Configuration:")
    print(f"  Exploration (1-probGreedy): {1-probGreedy}")
    print(f"  Learning Rate: {learningRate}")
    print(f"  Network Architecture: {layer_sizes}")
    print(f"  Regularization: {regularization}")
    
    rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=maxTrajectoryLen)
    
    # Calculate statistics
    avg_reward = np.mean(rewards)
    final_10_avg = np.mean(rewards[-10:])
    
    print(f"\nResults:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Final 10 Epochs Average: {final_10_avg:.2f}")
    print(f"  Best Epoch Reward: {max(rewards):.2f}")
    print(f"  Worst Epoch Reward: {min(rewards):.2f}")
    
    return rewards, agent

def run_improved_config_v1():
    """Configuration based on Task 2 and Task 3 insights"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Improved Config v1 (Better Exploration + LR)")
    print("="*60)
    
    seedValue = 42
    seed(seedValue)
    
    numStates = 45
    numActions = 15
    start = 0
    mdp = generate_random_mdp(numStates, numActions)
    
    # Improved based on Task 2 (exploration) and Task 3 (learning rate)
    probGreedy = .8  # Less exploration (1-0.8=0.2 like Task 2 found optimal)
    discountFactor = .9
    learningRate = .01  # Higher LR like Task 3 showed works well
    layer_sizes = [128]
    regularization = .0001
    
    agent = DeepQNetworkAgent("Improved_v1_DQN", numActions, mdp.numStates, probGreedy, discountFactor,
                              learningRate, layer_sizes, regularization, seedValue, Verbosity.SILENT)
    
    numSamples = 50
    numEpochs = 50
    maxTrajectoryLen = 100
    
    print(f"Configuration:")
    print(f"  Exploration (1-probGreedy): {1-probGreedy} (reduced based on Task 2)")
    print(f"  Learning Rate: {learningRate} (increased based on Task 3)")
    print(f"  Network Architecture: {layer_sizes}")
    print(f"  Regularization: {regularization}")
    
    rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=maxTrajectoryLen)
    
    avg_reward = np.mean(rewards)
    final_10_avg = np.mean(rewards[-10:])
    
    print(f"\nResults:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Final 10 Epochs Average: {final_10_avg:.2f}")
    print(f"  Best Epoch Reward: {max(rewards):.2f}")
    print(f"  Worst Epoch Reward: {min(rewards):.2f}")
    
    return rewards, agent

def run_improved_config_v2():
    """Configuration adding Task 4 insights (network depth)"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Improved Config v2 (+ Better Architecture)")
    print("="*60)
    
    seedValue = 42
    seed(seedValue)
    
    numStates = 45
    numActions = 15
    start = 0
    mdp = generate_random_mdp(numStates, numActions)
    
    # Adding Task 4 insight (more layers)
    probGreedy = .8
    discountFactor = .9
    learningRate = .01
    layer_sizes = [128, 64, 32]  # 3 layers like Task 4 showed works well
    regularization = .0001
    
    agent = DeepQNetworkAgent("Improved_v2_DQN", numActions, mdp.numStates, probGreedy, discountFactor,
                              learningRate, layer_sizes, regularization, seedValue, Verbosity.SILENT)
    
    numSamples = 50
    numEpochs = 50
    maxTrajectoryLen = 100
    
    print(f"Configuration:")
    print(f"  Exploration (1-probGreedy): {1-probGreedy}")
    print(f"  Learning Rate: {learningRate}")
    print(f"  Network Architecture: {layer_sizes} (3 layers based on Task 4)")
    print(f"  Regularization: {regularization}")
    
    rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=maxTrajectoryLen)
    
    avg_reward = np.mean(rewards)
    final_10_avg = np.mean(rewards[-10:])
    
    print(f"\nResults:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Final 10 Epochs Average: {final_10_avg:.2f}")
    print(f"  Best Epoch Reward: {max(rewards):.2f}")
    print(f"  Worst Epoch Reward: {min(rewards):.2f}")
    
    return rewards, agent

def run_optimal_config():
    """Final optimal configuration based on all tasks"""
    print("\n" + "="*60)
    print("EXPERIMENT 4: OPTIMAL Configuration (All Insights)")
    print("="*60)
    
    seedValue = 42
    seed(seedValue)
    
    numStates = 45
    numActions = 15
    start = 0
    mdp = generate_random_mdp(numStates, numActions)
    
    # Optimal config based on Tasks 1-6
    probGreedy = .8      # Task 2: moderate exploration
    discountFactor = .9
    learningRate = .01   # Task 3: aggressive but stable
    layer_sizes = [100, 100, 50]  # Task 4 & 5: depth + width balance
    regularization = 0.0  # Task 6: no regularization
    
    agent = DeepQNetworkAgent("OPTIMAL_DQN", numActions, mdp.numStates, probGreedy, discountFactor,
                              learningRate, layer_sizes, regularization, seedValue, Verbosity.SILENT)
    
    numSamples = 50
    numEpochs = 50
    maxTrajectoryLen = 100
    
    print(f"Configuration:")
    print(f"  Exploration (1-probGreedy): {1-probGreedy} (Task 2)")
    print(f"  Learning Rate: {learningRate} (Task 3)")
    print(f"  Network Architecture: {layer_sizes} (Tasks 4 & 5)")
    print(f"  Regularization: {regularization} (Task 6)")
    
    rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=maxTrajectoryLen)
    
    avg_reward = np.mean(rewards)
    final_10_avg = np.mean(rewards[-10:])
    
    print(f"\nResults:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Final 10 Epochs Average: {final_10_avg:.2f}")
    print(f"  Best Epoch Reward: {max(rewards):.2f}")
    print(f"  Worst Epoch Reward: {min(rewards):.2f}")
    
    return rewards, agent

def compare_all_configs():
    """Run all configurations and compare"""
    
    print("\n" + "="*70)
    print("TASK 7: OPTIMAL DQN DESIGN - COMPARATIVE EXPERIMENTS")
    print("="*70)
    print("\nRunning 4 different configurations to find the best DQN design...\n")
    
    # Run all experiments
    baseline_rewards, baseline_agent = run_baseline_config()
    improved_v1_rewards, improved_v1_agent = run_improved_config_v1()
    improved_v2_rewards, improved_v2_agent = run_improved_config_v2()
    optimal_rewards, optimal_agent = run_optimal_config()
    
    # Create comparison visualization
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    plt.figure(figsize=(14, 8))
    
    # Learning curves
    plt.subplot(1, 2, 1)
    epochs = range(50)
    plt.plot(epochs, baseline_rewards, 'r-', linewidth=2, alpha=0.7, label='Baseline (Default)')
    plt.plot(epochs, improved_v1_rewards, 'orange', linewidth=2, alpha=0.7, label='Improved v1 (Exploration+LR)')
    plt.plot(epochs, improved_v2_rewards, 'b-', linewidth=2, alpha=0.7, label='Improved v2 (+Architecture)')
    plt.plot(epochs, optimal_rewards, 'g-', linewidth=3, alpha=0.8, label='OPTIMAL (All Insights)')
    plt.xlabel('Training Epoch')
    plt.ylabel('Average Reward')
    plt.title('Task 7: DQN Configuration Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistical comparison
    plt.subplot(1, 2, 2)
    configs = ['Baseline', 'Improved v1', 'Improved v2', 'OPTIMAL']
    avg_rewards = [np.mean(baseline_rewards), np.mean(improved_v1_rewards), 
                   np.mean(improved_v2_rewards), np.mean(optimal_rewards)]
    final_rewards = [np.mean(baseline_rewards[-10:]), np.mean(improved_v1_rewards[-10:]),
                     np.mean(improved_v2_rewards[-10:]), np.mean(optimal_rewards[-10:])]
    
    x = np.arange(len(configs))
    width = 0.35
    
    plt.bar(x - width/2, avg_rewards, width, label='Overall Avg', alpha=0.8)
    plt.bar(x + width/2, final_rewards, width, label='Final 10 Epochs', alpha=0.8)
    plt.xlabel('Configuration')
    plt.ylabel('Average Reward')
    plt.title('Performance Comparison')
    plt.xticks(x, configs, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('task7_optimal_dqn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\nSUMMARY OF CONFIGURATIONS:")
    print("-" * 60)
    for i, (config, avg, final) in enumerate(zip(configs, avg_rewards, final_rewards)):
        print(f"{config}:")
        print(f"  Overall Average: {avg:.2f}")
        print(f"  Final Performance: {final:.2f}")
        improvement = ((avg - avg_rewards[0]) / abs(avg_rewards[0])) * 100 if avg_rewards[0] != 0 else 0
        print(f"  Improvement over Baseline: {improvement:+.1f}%")
        print()
    
    # Determine winner
    best_idx = np.argmax(final_rewards)
    print(f"🏆 WINNER: {configs[best_idx]} with {final_rewards[best_idx]:.2f} final performance!")
    
    return optimal_agent

if __name__ == "__main__":
    print("Starting Task 7: Optimal DQN Design Experiments...")
    print("This will take a few minutes to run all 4 configurations...\n")
    
    best_agent = compare_all_configs()
    
    print("\n✅ Task 7 Complete! Best DQN configuration identified.")
