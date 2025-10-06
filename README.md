Prerequisites

Python 3.7 or higher
Required Python packages:
gymnasium>=0.29.1
numpy>=1.21.0
matplotlib>=3.5.0



Installation

Clone or download the project repository to your local machine.
Install the required dependencies using pip:pip install gymnasium numpy matplotlib




The script will:
Train and evaluate five reinforcement learning agents (Random, Value Iteration, Policy Iteration, Monte Carlo, Q-Learning) on a 5x5 GridWorld.
Display performance metrics (mean rewards, steps, and training time) for each agent.
Generate a matplotlib figure comparing agent performance (rewards, steps, training time, and a summary table).
Prompt you to select an agent (1-5) to simulate its movement step-by-step, showing the grid, actions, rewards, and total steps.



Notes

Ensure you have a terminal that supports Unicode characters for proper display of the grid (e.g., A for agent, G for goal).
The simulation clears the terminal/console for each step (works on Windows and Unix-based systems).
If you encounter issues, verify that all dependencies are installed correctly and Python is version 3.7 or higher.
