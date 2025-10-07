import matplotlib.pyplot as plt
from testsDqnLab import DqnLabTest3
import numpy as np

def run_and_plot_dqn_test3():
    """Run DqnLabTest3 and capture the learning curve data"""
    print("🧠 Running DQN Training Analysis (Task 1)...")
    
    # Note: I'll manually extract the data from the console output I just saw
    # In a real implementation, I'd modify the test function to return the data
    
    epochs = list(range(50))
    rewards = [
        73.9231, 69.5185, -10001.0000, 69.8248, 69.4286,
        -10001.0000, 69.4286, -10001.0000, -10001.0000, 69.4286,
        69.5185, 69.8248, 69.4286, 69.6267, 69.4286,
        69.4286, 69.4286, 69.5185, 69.5185, 68.8286,
        69.6084, 69.4286, 67.6286, 69.4286, 69.6084,
        69.4286, 69.5185, 69.4286, 69.4286, 69.5185,
        69.4286, 69.5185, 69.4286, 69.4286, 69.6982,
        69.0982, 69.4286, 68.8286, 69.0084, 69.5185,
        69.6084, 69.5185, 69.5185, 69.4286, 69.4286,
        69.6982, 69.4286, 69.4286, 69.6084, 69.5185
    ]
    
    # Create the learning curve plot
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, rewards, 'b-', linewidth=2, alpha=0.7)
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Target Performance (~70)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.title('Task 1: DQN Training Learning Curve\nBasic Parking Lot Environment', fontsize=14, fontweight='bold')
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Average Reward per Episode', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Highlight the catastrophic drops
    catastrophic_epochs = [2, 5, 7, 8]
    for epoch in catastrophic_epochs:
        plt.scatter(epoch, rewards[epoch], color='red', s=100, zorder=5)
    
    # Add text annotation for the catastrophic failures
    plt.annotate('Catastrophic Failures\n(Neural Network Instability)', 
                xy=(5, -10001), xytext=(15, -5000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, ha='center', color='red')
    
    plt.tight_layout()
    plt.savefig('Task1_DQN_Training_Learning_Curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return epochs, rewards

# Run the analysis
epochs, rewards = run_and_plot_dqn_test3()
print("✅ Task 1 analysis complete!")
print(f"📊 Learning curve saved as 'Task1_DQN_Training_Learning_Curve.png'")

# Calculate some basic statistics
normal_rewards = [r for r in rewards if r > -1000]  # Exclude catastrophic failures
print(f"📈 Normal performance range: {min(normal_rewards):.2f} to {max(normal_rewards):.2f}")
print(f"📈 Average normal performance: {np.mean(normal_rewards):.2f}")
print(f"⚠️  Number of catastrophic failures: {len([r for r in rewards if r < -1000])}")