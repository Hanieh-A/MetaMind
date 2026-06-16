# MetaMind: LLM-Orchestrated Computational Intelligence Framework

MetaMind is a Computational Intelligence framework that combines Large Language Models (LLMs) with classical and modern CI techniques. The system acts as an intelligent orchestrator that analyzes a problem, selects an appropriate computational intelligence method, configures its parameters, executes the algorithm, evaluates the results, and provides recommendations for improvement.

This project was developed as the final project for a Computational Intelligence course.

---

## Overview

Traditional computational intelligence systems require users to manually choose algorithms and tune parameters. MetaMind introduces an LLM-based orchestration layer that automates this process.

The framework supports multiple problem domains:

* Optimization
* Classification
* Clustering

The orchestrator can:

1. Analyze the problem description.
2. Identify the problem type.
3. Select a suitable CI algorithm.
4. Generate algorithm parameters.
5. Execute the selected method.
6. Evaluate performance.
7. Provide natural-language explanations and recommendations.

---

## Architecture

```text
User Problem
      │
      ▼
┌──────────────────┐
│ LLM Orchestrator │
└────────┬─────────┘
         │
         ├── Problem Analysis
         ├── Method Selection
         ├── Parameter Configuration
         ▼
┌──────────────────┐
│ CI Method Layer  │
└────────┬─────────┘
         │
         ▼
  Algorithm Execution
         │
         ▼
   Evaluation Engine
         │
         ▼
 Result Interpretation
```

---

## Implemented Computational Intelligence Methods

### Neural Networks

* Perceptron
* Multi-Layer Perceptron (MLP)
* Kohonen Self-Organizing Map (SOM)
* Hopfield Network

### Fuzzy Systems

* Fuzzy Controller

### Evolutionary & Swarm Intelligence

* Genetic Algorithm (GA)
* Genetic Programming (GP)
* Particle Swarm Optimization (PSO)
* Ant Colony Optimization (ACO)

---

## Supported Problems

### 1. Traveling Salesman Problem (TSP)

Graph-based combinatorial optimization problem solved using methods such as:

* Ant Colony Optimization
* Genetic Algorithm

Metrics:

* Tour Length
* Gap to Optimal
* Computation Time
* Success Rate

---

### 2. Function Optimization

Benchmark optimization functions:

* Sphere
* Rastrigin
* Ackley
* Rosenbrock

Metrics:

* Best Fitness
* Mean Fitness
* Standard Deviation
* Error
* Success Rate

---

### 3. Titanic Survival Prediction

Binary classification problem using machine learning and neural network approaches.

Tasks:

* Data preprocessing
* Feature encoding
* Missing value handling
* Model training and evaluation

Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

### 4. Clustering

Unsupervised learning experiments on:

* Iris Dataset
* Mall Customer Dataset
* Synthetic Datasets

Metrics:

* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Index
* Adjusted Rand Index
* Normalized Mutual Information

---

## LLM Orchestrator

The orchestrator is responsible for:

* Problem parsing
* Method selection
* Parameter generation
* Execution management
* Result interpretation
* Recommendation generation

Example output:

```json
{
  "problem_type": "optimization",
  "selected_method": "PSO",
  "confidence": 0.89,
  "parameters": {
    "n_particles": 50,
    "max_iterations": 500
  }
}
```

---

## Features

* Unified interface for all CI methods
* Automatic algorithm selection
* Automatic parameter configuration
* Result analysis and interpretation
* Convergence tracking
* Experiment logging
* Comparative evaluation
* Extensible architecture for adding new algorithms

---

## Project Structure

```text
MetaMind/
│
├── src/
│   ├── methods/
│   ├── orchestrator/
│   ├── problems/
│   └── evaluation/
│
├── experiments/
├── datasets/
├── results/
├── reports/
└── README.md
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* OpenAI API / LLM Integration

---

## Future Improvements

* Multi-method ensemble orchestration
* Automatic hyperparameter optimization
* Reinforcement-learning-based method selection
* Experiment memory and continual learning
* Distributed execution for large-scale experiments

---

## Course Information

**Course:** Computational Intelligence

**Project:** MetaMind – LLM-Orchestrated Computational Intelligence Framework

This project explores how Large Language Models can serve as intelligent controllers for computational intelligence systems, enabling automated algorithm selection, configuration, evaluation, and recommendation generation.
