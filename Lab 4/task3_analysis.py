#!/usr/bin/env python3
"""
Task 3: Learning Rate Analysis
Running DqnLabTest5 to compare three different learning rates
"""

import sys
sys.path.append('.')

from testsDqnLab import DqnLabTest5

if __name__ == "__main__":
    print("Starting Task 3: Learning Rate Analysis...")
    print("Running DqnLabTest5 with three different learning rates...")
    print()
    
    # Run the test with seed 1
    DqnLabTest5(1)
    
    print("\n✅ Task 3 Complete!")
    print("The plot window shows learning curves for three different learning rates:")
    print("  - HIGH (0.01): Faster initial learning but potential instability")
    print("  - MED (0.0001): Balanced convergence speed and stability")
    print("  - LOW (0.00001): Slow but very stable learning")
