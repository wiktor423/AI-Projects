# Pathfinding Search Algorithms

This project implements and visualizes heuristic search algorithms for pathfinding in a 2D grid maze. It features a fully-fledged console visualization framework that animates the step-by-step grid exploration of the agent.

## Core Features

- **Greedy Best-First Search**: An informed search algorithm that expands nodes based on the closest estimated distance to the target.
- **Supported Heuristics**:
  - **Manhattan Distance ($L_1$ norm)**: Best suited for 4-directional grid movements.
  - **Euclidean Distance ($L_2$ norm)**: Measures straight-line distance, which never overestimates the path but is less informed than Manhattan on grid constraints.
  - **Chebyshev Distance ($L_\infty$ norm)**: Useful when 8-directional moves (including diagonals) are allowed.
- **Terminal Visualization**: Displays step-by-step search progression, including currently expanded cells, visited cells, walls, and the final reconstructed path.
- **Comprehensive Test Cases**: Features 12 unit tests verifying edge cases, straight corridors, start/finish alignment, dead ends, and blocked paths.

## Directory Structure

- [main.py](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/01-pathfinding-search/main.py): Contains search implementation, heuristics, ASCII animation suite, and testing suite.
- [lab1_cg105_g27_v2_Klepacz_Savy.pdf](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/01-pathfinding-search/lab1_cg105_g27_v2_Klepacz_Savy.pdf): Project report detailing performance comparisons.

## Setup and Usage

### Requirements
- Python 3.x (no external packages are required for pathfinding as it uses standard libraries: `heapq`, `math`, `os`, `time`).

### How to Run
Execute the script from the command line:
```bash
python main.py
```
This will run all 12 test cases, output a comparative summary table of path lengths and cell expansions for each heuristic, and then launch an interactive terminal animation showing the search process.

## Visualization Key
- `██` : Wall / Blocked cell
- `  ` : Open cell
- `S ` : Start position
- `F ` : Finish position
- `@ ` : Currently expanding cell
- `· ` : Visited cell
- `▪ ` : Final reconstructed path

## Authors
- **Savy Timothée**
- **Klepacz Wiktor**
- **Group Number**: 27
