# Studying the evolution of altruistic behavior via Genetic Algorithms

This repository contains the source code for the project for the game theory course at ETH. Our study is focussed on the paper "Is altruism evolutionary stable?" [[1]](#1). The `main` branch consists of the following files:

1. `agent.py` initializes an agent object for the simulation
2. `genetic_algorithm.py` defines a class that implements the genetic algorithm
3. `arguments.py` details the arguments passed in the command line
4. `setup.py` sets up the module for the project
5. The `experiments` folder consists of the following:
  a) `plots.py` plotting functions for the outputs
  b) `variation_1_and_2.py` refers to the simulation of the egoistic equilibrium with mutations in strategy
  c) `variation_3.py` refers to the game with fixed altruism
  d) `variation_4.py` allows the mutability of the altruism preference parameter
  e) `prop1_and_2.py` uses mutable altruism to prove the propositions 1 and 2 from [[1]](#1)

## References
<a id="1">[1]</a> 
Bester, H. and Güth, W., 1998. 
Is altruism evolutionary stable?
*Journal of Economic Behavior & Organization*, 34(2), pp.193-209.
