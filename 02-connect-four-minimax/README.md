# Connect Four Minimax AI

An implementation of the Connect Four game featuring an AI opponent built on a minimax decision tree with alpha-beta pruning.

## Core Features

- **Minimax Algorithm**: Evaluates future move trees up to a specified depth (default depth of 6 for live play) to determine optimal moves.
- **Alpha-Beta Pruning**: Significantly optimizes the search by pruning branches that are guaranteed to be worse than previously evaluated moves.
- **Heuristic Position Evaluation**:
  - **Window Evaluation**: Evaluates all horizontal, vertical, and diagonal windows of 4 cells. Assigns scores based on player/opponent piece distribution (highly penalizes allowing opponent 3-in-a-row).
  - **Column Control**: Prefers placing pieces in the center column (+3 per piece) and adjacent columns (+2 per piece) to establish early board control.
- **Unit Testing Suite**: Features 6 automated test scenarios verifying AI capabilities, such as taking winning moves, blocking user paths (vertical, horizontal, and gaps), and prioritizing winning over blocking.
- **Interactive Gameplay**: Command-line interface allowing the user to select player order (first or second) and play against the AI.

## Directory Structure

- [main.py](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/02-connect-four-minimax/main.py): Contains the complete game logic, minimax algorithm, evaluation functions, unit tests, and terminal game loop.

## Setup and Usage

### Requirements
- Python 3.x (uses only Python standard library: `copy`, `math`, `random`).

### How to Run
Run the script using Python:
```bash
python main.py
```
Upon execution, you will be prompted:
1. To choose whether you want to go first (`1` for 'X') or second (`2` for 'O').
2. Whether to run the automated AI behavior test cases before the game starts.

## Evaluation Scoring System
The AI computes board state quality using the following weights:
- **4 AI pieces**: $+1,000,000$ (Terminal Win)
- **4 Player pieces**: $-1,000,000$ (Terminal Loss)
- **3 AI pieces + 1 empty**: $+10$
- **3 Player pieces + 1 empty**: $-80$ (Priority block)
- **2 AI pieces + 2 empty**: $+2$
- **2 Player pieces + 2 empty**: $-10$
- **Center Column piece**: $+3$
- **Column 2 & 4 pieces**: $+2$

## Authors
- **Savy Timothée**
- **Klepacz Wiktor**
- **Group Number**: 27
