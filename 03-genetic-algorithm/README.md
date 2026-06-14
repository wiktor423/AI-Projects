# Genetic Algorithm Optimization

This project implements a Genetic Algorithm (GA) from scratch in Python to find the global minimum of the 2D **Himmelblau's function**:
$$f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$$

## Core Features

- **Binary Chromosome Representation**: Converts binary strings into real numbers within the Area of Interest (AOI) $[-5, 5]$ using min-max mapping.
- **Crossover & Mutation**:
  - **One-point Crossover**: Randomly selects a crossover point and swaps genetic material between parents.
  - **Single-gene Mutation**: Flips individual bits with a set probability to introduce random diversity and escape local minima.
- **Tournament Selection**: Selects parent chromosomes for the next generation based on tournament competition (default tournament size: 2).
- **Hyperparameter Tuning Study**: Conducts systematic studies on crucial parameters, running 7 trials for each configuration, and outputting their performance (mean and standard deviation). Evaluates:
  - Mutation Probability ($0.01, 0.05, 0.1, 0.3$)
  - Tournament Size ($2, 5, 10, 20$)
  - Population Size ($20, 50, 100, 200$)
  - Crossover Probability ($0.4, 0.6, 0.8, 1.0$)
- **Trace Visualization**: Visualizes optimization trajectory over time by plotting individual generation bests directly on a contour map of Himmelblau's function using `matplotlib`.

## Directory Structure

- [main.py](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/03-genetic-algorithm/main.py): Primary codebase defining the `Chromosome` and `GeneticAlgorithm` classes, the parameter tuning suite, and visualization logic.
- [assignment_3.md](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/03-genetic-algorithm/assignment_3.md): Project instructions or task descriptions.
- [lab3_cg_105_g27_v2_Klepacz_Savy.pdf](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/03-genetic-algorithm/lab3_cg_105_g27_v2_Klepacz_Savy.pdf): Final report documenting experimental results and analysis.

## Setup and Usage

### Requirements
Ensure you have the required dependencies installed:
```bash
pip install numpy matplotlib
```

### How to Run
Run the parameter study and visualization:
```bash
python main.py
```
This runs the full suite of parameter optimization experiments, prints statistical summaries of each run to the terminal, and displays matplotlib contour plots showing the path of the algorithm towards the function's minima.

## Authors
- **Savy Timothée**
- **Klepacz Wiktor**
- **Group Number**: 27
