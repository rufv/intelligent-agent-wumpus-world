# intelligent-agent-wumpus-world
Autonomous AI agent for the Wumpus World environment. Implements pathfinding (BFS), risk assessment logic, and state-space exploration under partial observability.

---

# AI Intelligent Agent: Wumpus World Solver

This project features an autonomous **Intelligent Agent** designed to navigate the classic "Wumpus World" environment, a standard AI benchmark for reasoning under uncertainty.

## Agent Intelligence (Implemented in `solution.py`)

The agent is designed to collect gold and escape while avoiding pits and the deadly Wumpus. Key implemented features include:

- **Logic-Based Exploration:** Tracks `visited`, `safe`, and `frontier` cells to make safe movement decisions.
- **Perception Processing:** Analyzes `Breeze` and `Stench` signals to dynamically update internal risk maps (Pit/Wumpus suspicion counts).
- **Optimal Pathfinding:** Uses **Breadth-First Search (BFS)** to calculate the shortest path between the agent's position and safe targets or the exit.
- **Dynamic Action Planning:** Utilizes a `deque`-based planning system to execute multi-step movement sequences (Rotate + Move).
- **Failure Safety:** Implements a "Return Home" strategy if no further safe exploration is possible.

## Project Structure

- `solution.py`: **My core implementation** of the Agent's decision-making logic.
- `wumpus.py`: The environment simulator (grid physics, sensors, and rewards).
- `interact.py`: Simulation runner to evaluate agent performance over 1000 episodes.
**Note:** The `icons/` directory is not included in this repository to respect the original creators' licensing. 
The core logic in `solution.py` can be evaluated using the `interact.py` script which runs simulations in the console.

## Technical Stack
- **Language:** Python 3.10+
- **Algorithms:** BFS (Shortest Path), State-Space Search.
- **Concepts:** Partially Observable Markov Decision Processes (POMDP) basics, Heuristic Reasoning.

## How to Run
```bash
python interact.py
```

## Credits
Environment & Framework: Provided by the University of Vienna (Foundations of Intelligent Systems course).
Agent Logic: Fully implemented by me.
