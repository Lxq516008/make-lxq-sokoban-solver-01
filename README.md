# Sokoban Solver

[![make.fun](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/Lxq516008/make-lxq-sokoban-solver-01/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/Lxq516008/make-lxq-sokoban-solver-01/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver for [make.fun](https://make.fun/bounty/sokoban-solver-01-run-20261121): one `main.ts` (Node 22, no dependencies) that reads each `level.txt` and writes `solution.txt`.

How it works: the level is a hallway with rooms off it, and a box can never leave its room, so the level splits into independent rooms. Each room is solved on its own with A* over pushes (boxes + player region as the state, cheapest box-to-goal assignment as the estimate, dead-square and frozen-block pruning), with greedy and breadth-first fallbacks. Practice set: 1,000,000/1,000,000 in ~0.4 s; house baseline scores 200,000.
