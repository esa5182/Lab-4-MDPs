# Lab 4 - Deep Q-Network (DQN)

## Overview

This repository contains the implementation and analysis for Lab 4, exploring Deep Q-Networks (DQN) for reinforcement learning in discrete Markov Decision Processes.

## Files Included

### Core Framework
- `testsDqnLab.py` - Test framework with DqnLabTest1-10
- `MDP.py` - Markov Decision Process environment
- `ParkingDefs.py` - Parking lot environment definitions
- `ParkingLotFactory.py` - Environment generator
- `main.py` - Main execution script
- `agent/` - DQN agent implementation
- `data/` - MDP configuration files (MDP1.txt, MDP2.txt, MDP3.txt)

### Task Analysis Scripts
- `task1_analysis.py` - Task 1: DQN training analysis (DqnLabTest3)
- `task2_analysis.py` - Task 2: Greed comparison (DqnLabTest4)
- `task3_analysis.py` - Task 3: Learning rate comparison (DqnLabTest5)
- `task4_analysis.py` - Task 4: Network depth analysis (DqnLabTest7)
- `task5_analysis.py` - Task 5: Network width analysis (DqnLabTest8)
- `task6_analysis.py` - Task 6: Regularization analysis (DqnLabTest9)
- `task7_optimal_dqn.py` - Task 7: Optimal DQN design (DqnLabTest10)
- `task8_comparative_analysis.py` - Task 8: DQN vs Q-Learning comparison

### Generated Charts
- `Task1_DQN_Training_Learning_Curve.png`
- `Task2_Greed_Comparison.png`
- `Task3_Learning_Rate_Comparison.png`
- `task4_network_depth_analysis.png`
- `task5_network_width_analysis.png`
- `task6_regularization_analysis.png`
- `task7_optimal_dqn_comparison.png`
- `task8_dqn_vs_qlearning_comparison.png`

### Submission Document
- `Lab4_Submission.md` - Complete written answers to all 10 tasks

## Requirements

```
numpy
scikit-learn
matplotlib
```

Install dependencies:
```bash
pip install numpy scikit-learn matplotlib
```

## Running the Tasks

### Method 1: Run tests directly
```bash
python testsDqnLab.py 3    # Task 1 - DQN Training (DqnLabTest3)
python testsDqnLab.py 4    # Task 2 - Greed Analysis (DqnLabTest4)
python testsDqnLab.py 5    # Task 3 - Learning Rate (DqnLabTest5)
python testsDqnLab.py 7    # Task 4 - Network Depth (DqnLabTest7)
python testsDqnLab.py 8    # Task 5 - Network Width (DqnLabTest8)
python testsDqnLab.py 9    # Task 6 - Regularization (DqnLabTest9)
python testsDqnLab.py 10   # Task 7 - Optimal Design (DqnLabTest10)
```

### Method 2: Run analysis scripts (generates charts)
```bash
python task1_analysis.py   # Generates Task1_DQN_Training_Learning_Curve.png
python task2_analysis.py   # Generates Task2_Greed_Comparison.png
python task3_analysis.py   # Generates Task3_Learning_Rate_Comparison.png
python task4_analysis.py   # Generates task4_network_depth_analysis.png
python task5_analysis.py   # Generates task5_network_width_analysis.png
python task6_analysis.py   # Generates task6_regularization_analysis.png
python task7_optimal_dqn.py # Generates task7_optimal_dqn_comparison.png
python task8_comparative_analysis.py # Generates task8_dqn_vs_qlearning_comparison.png
```

## Submission Contents

As per assignment requirements, this repository includes:
1. **Charts/Learning curves** - All required visualizations for Tasks 1-8
2. **Explanations** - Complete written analysis in `Lab4_Submission.md`
3. **Neural network creation function** - Optimal DQN design in `task7_optimal_dqn.py`

## Key Findings

- Deep Q-Networks show higher training volatility compared to tabular Q-learning
- Network architecture significantly impacts training stability
- For small discrete state spaces (45 states), simpler networks perform better
- DQN's advantages emerge primarily in large-scale or continuous state problems
- Optimal configuration: 1 hidden layer, 32 neurons, learning rate 0.01, exploration 0.3

## Assignment Details

**Course:** DS-402 Trends in Data Science  
**Lab:** Lab 4 - Training a Deep Q-Network

See `Lab4_Submission.md` for complete experimental analysis and results.
