# Hive Implementation in python

This repository contains a Python implementation of the Hive board game, featuring a Tkinter GUI for ease of use. Players can either play against a friend or face a basic "AI" opponent that makes random legal moves. The game follows the standard Hive rules, and the GUI allows for easy interaction with the game board.

![Hive](https://user-images.githubusercontent.com/74073756/229519349-4fac5067-646c-416c-a046-c1c91a0b2516.gif)

## Installation

1. Clone the repository
```bash
git clone https://github.com/your_username/hive-tkinter.git
```
2. Navigate to the repository folder
```bash
cd hive-tkinter
```
3. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage

Start the game by running 
```Hive_GUI.py```

```bash
python Hive_GUI.py
```

The game window will open, showing the Hive board.

Players can take turns placing and moving their pieces using the on-screen buttons and clicking on the board.

The game will continue until one player wins by surrounding the opponent's Queen Bee, or until a draw is declared.

## Game Rules

For a detailed explanation of the game rules, please refer to the official Hive rules.

## Project Structure

The project is organized as follows:

- src: Contains the source code for the game implementation
- scripts: contain scripts for conducting experiments and testing the game
  - evaluate.py: Simulates the game between two AI players and evaluates their performance
  - LearnHive: first approach to implement a reinforcement learning agent with Monte Carlo Tree Search (under review)
  - profile_hive.py: Profiles the Hive game to identify performance bottlenecks. (open issue)
  - random_game.py: Simulates a random game between two AI players and creates a gif of the game
- tests: Contains unit tests for the game implementation
- old: Contains old versions of the game implementation (to be removed in the future)

## Future Plans

The current AI opponent makes random legal moves. In the future, we plan to implement an AI opponent using reinforcement learning techniques, which will make the AI more challenging and engaging for players.Contributing

Pull requests are welcome. Please open an issue to discuss any major changes you would like to make.

## License

This project is licensed under the MIT License. See the LICENSE file for details.