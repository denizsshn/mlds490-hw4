# Appendix – Instructions for Running the Code

## Requirements

To run this code, you need Python 3 and the following libraries: `numpy`, `torch`, `scikit-learn`, `bayesian-optimization`, and `matplotlib`.

You can install all requirements using pip:

```bash
pip install numpy torch scikit-learn bayesian-optimization matplotlib
```

## Data Setup

The script expects the Federated EMNIST dataset files to be located in the same directory as the python script. It supports loading from either the raw dictionary format or pre-split arrays. Ensure the following files are present:

- `train_data.npy`
- `test_data.npy`

(Note: The script automatically filters the dataset to include only the 10 digit classes upon loading.)

## File Structure and Execution

The entire project is contained within a single Python script. The script is structured into modular classes and functions that execute the Genetic Algorithm and Bayesian Optimization sequentially.

### Classes and Model Definition

- `SimpleClassifier`: The neural network architecture used for the task.
  - A 2-layer fully connected network.
  - Structure: Input (Dynamic) → Hidden (256) → Activation → Output (10).
  - The activation function is dynamically set based on the hyperparameter search (`ReLU`, `Sigmoid`, or `Tanh`).

- `GeneticAlgorithm`: A custom class implementing the evolutionary search.
  - Implements `roulette_selection` for choosing parents based on fitness.
  - Manages the population evolution with Elitism (preserving the best individual).
  - Tracks average and best fitness scores across generations.

- `load_data`: A utility function for preprocessing.
  - Flattens `28×28` images into vectors.
  - Normalizes pixel values to `[0, 1]`.
  - Filters the dataset to retain only digits (labels `0–9`) and performs an 80/20 train-validation split.

### Execution Flow

The `main` execution block orchestrates the experiments in the following order:

- **Phase 1: Genetic Algorithm Search**
  - Initializes a population of 20 individuals with random hyperparameters.
  - Evolves the population for 20 generations.
  - Calculates fitness using a reduced training duration (10 epochs).
  - Saves the progress plot as `ga_progress.png`.

- **Phase 2: Bayesian Optimization Search** (`run_bayesian_optimization`)
  - Uses the `BayesianOptimization` package to maximize the validation F1 score.
  - Runs for 50 total iterations (10 random initialization points + 40 optimization steps).
  - Outputs the best hyperparameter combination found.

- **Phase 3: Final Evaluation** (`final_train_and_report`)
  - Takes the best hyperparameters from GA and BO.
  - Merges training and validation sets for maximum data utility.
  - Trains two separate models from scratch for 50 epochs.
  - Generates training curves (`training_curve_GA.png`, `training_curve_BO.png`) and reports the final Test F1 scores.

## How to Run

Simply execute the script from the command line:

```bash
python hw4.py
```

The script will automatically detect if a GPU (CUDA) is available and utilize it for training. All plots and metrics will be saved to the current working directory upon completion.
```
