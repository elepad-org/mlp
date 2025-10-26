"""Multilayer Perceptron for pattern recognition.

This module contains the MLP implementation extracted from the notebook,
ready for production use with model saving/loading capabilities.
"""

import numpy as np
import pandas as pd
from typing import Literal, Dict
import pickle
from datetime import datetime
from pathlib import Path


# Expected outputs for each pattern
EXPECTED_OUTPUT = {
    "b": np.array([1, 0, 0]),
    "d": np.array([0, 1, 0]),
    "f": np.array([0, 0, 1]),
}


class MLP:
    """Multilayer Perceptron for pattern recognition."""

    def __init__(
        self,
        activation_type: Literal["sigmoid", "linear"] = "sigmoid",
        learning_rate: float = 0.1,
        momentum: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize the MLP with specified hyperparameters.

        Args:
            activation_type: Activation function type ("sigmoid" or "linear")
            learning_rate: Learning rate for gradient descent
            momentum: Momentum coefficient for parameter updates
            seed: Random seed for reproducibility
        """
        self.activation_type = activation_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Training history
        self.history = {
            "train_losses": [],
            "val_losses": [],
            "epochs": 0,
        }

        # Model metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "architecture": {
                "input_size": 100,
                "hidden1_size": 10,
                "hidden2_size": 5,
                "output_size": 3,
            },
        }

        self.params = self.initialize_parameters()

    def initialize_parameters(self) -> Dict[str, np.ndarray]:
        """
        Initialize weights and biases using Xavier initialization, where each
        parameter will be drawn from a normal distribution with mean of `0` and
        a standard deviation of `sqrt(2 / (input_layer_size + output_layer_size))`
        to help prevent vanishing or exploding gradients.
        """
        # Initialize layer 1 parameters (input -> hidden1)
        std1 = np.sqrt(2.0 / (100 + 10))
        W1 = self.rng.normal(0, std1, (10, 100))
        b1 = np.zeros(10)

        # Initialize layer 2 parameters (hidden1 -> hidden2)
        std2 = np.sqrt(2.0 / (10 + 5))
        W2 = self.rng.normal(0, std2, (5, 10))
        b2 = np.zeros(5)

        # Initialize output layer parameters (hidden2 -> output)
        std3 = np.sqrt(2.0 / (5 + 3))
        W3 = self.rng.normal(0, std3, (3, 5))
        b3 = np.zeros(3)

        return {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3,
            "delta_W1": np.zeros_like(W1),
            "delta_b1": np.zeros_like(b1),
            "delta_W2": np.zeros_like(W2),
            "delta_b2": np.zeros_like(b2),
            "delta_W3": np.zeros_like(W3),
            "delta_b3": np.zeros_like(b3),
        }

    def get_activation_function(self):
        """Return the activation function based on activation type."""
        if self.activation_type == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation_type == "linear":
            return lambda x: x

    def get_activation_derivative(self):
        """Return the derivative of the activation function based on activation type."""
        if self.activation_type == "sigmoid":

            def sigmoid_derivative(x):
                s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
                return s * (1 - s)

            return sigmoid_derivative
        elif self.activation_type == "linear":
            return lambda x: np.ones_like(x)

    def feedforward(self, X: np.ndarray) -> list:
        """
        Perform a feedforward pass through the network.
        See: Prince, 2023 (page 103) or Hilera & Martinez, 2000 (pages 138-139).

        Args:
            X: Input pattern (100 values).
        Returns:
            List with all neuron activations at each layer [X, h1, h2, y].
        """
        a = self.get_activation_function()

        # Forward pass from input layer to hidden layer 1
        h1 = a(self.params["b1"] + np.dot(self.params["W1"], X))

        # Forward pass from hidden layer 1 to hidden layer 2
        h2 = a(self.params["b2"] + np.dot(self.params["W2"], h1))

        # Forward pass from hidden layer 2 to output layer
        y = a(self.params["b3"] + np.dot(self.params["W3"], h2))

        return [X, h1, h2, y]

    def backpropagation(self, X: np.ndarray, d: np.ndarray, activations: list) -> dict:
        """
        Backpropagation. See: Hilera & Martinez, 2000 (pages 133-142).

        Args:
            X: Input pattern (100 values).
            d: Target output (3 values).
            activations: Neuron activations from forward pass [X, h1, h2, y_pred].
        Returns:
            Dictionary with updated parameters.
        """
        _, h1, h2, y_pred = activations
        a_derived = self.get_activation_derivative()

        # Output layer (see page 134-135)
        delta3 = (d - y_pred) * a_derived(y_pred)

        delta_W3 = (
            self.learning_rate * np.outer(delta3, h2)
            + self.momentum * self.params["delta_W3"]
        )
        delta_b3 = self.learning_rate * delta3 + self.momentum * self.params["delta_b3"]

        # Hidden layer 2 (see page 135)
        delta2 = np.dot(self.params["W3"].T, delta3) * a_derived(h2)

        delta_W2 = (
            self.learning_rate * np.outer(delta2, h1)
            + self.momentum * self.params["delta_W2"]
        )
        delta_b2 = self.learning_rate * delta2 + self.momentum * self.params["delta_b2"]

        # Hidden layer 1 (see page 135)
        delta1 = np.dot(self.params["W2"].T, delta2) * a_derived(h1)

        delta_W1 = (
            self.learning_rate * np.outer(delta1, X)
            + self.momentum * self.params["delta_W1"]
        )
        delta_b1 = self.learning_rate * delta1 + self.momentum * self.params["delta_b1"]

        # Update parameters (see page 135)
        new_params = self.params.copy()
        new_params["W1"] += delta_W1
        new_params["b1"] += delta_b1
        new_params["W2"] += delta_W2
        new_params["b2"] += delta_b2
        new_params["W3"] += delta_W3
        new_params["b3"] += delta_b3

        # Update momentum terms (see page 135)
        new_params["delta_W1"] = delta_W1
        new_params["delta_b1"] = delta_b1
        new_params["delta_W2"] = delta_W2
        new_params["delta_b2"] = delta_b2
        new_params["delta_W3"] = delta_W3
        new_params["delta_b3"] = delta_b3

        return new_params

    def calculate_loss(self, d: np.ndarray, s: np.ndarray) -> float:
        """Compute least mean squared error loss. See: Hilera & Martinez, 2000 (page 119)."""
        return (1 / (2 * len(d))) * np.sum((d - s) ** 2)

    def classify(self, X: np.ndarray) -> str:
        """
        Classify the input pattern into a class.

        Args:
            X: Input pattern (100 values).
        Returns:
            "b", "d", or "f" according to the maximal output neuron.
        """
        y_pred = self.feedforward(X)[-1]
        idx = int(np.argmax(y_pred))
        classes = ["b", "d", "f"]
        return classes[idx]

    def predict_proba(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get prediction probabilities for each class.

        Args:
            X: Input pattern (100 values).
        Returns:
            Dictionary with probabilities for each class.
        """
        y_pred = self.feedforward(X)[-1]
        # Softmax to convert to probabilities
        exp_scores = np.exp(y_pred - np.max(y_pred))
        probabilities = exp_scores / np.sum(exp_scores)

        return {
            "b": float(probabilities[0]),
            "d": float(probabilities[1]),
            "f": float(probabilities[2]),
        }

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        tolerance: float = 1e-6,
        max_epochs: int = 10000,
        verbose: bool = True,
    ) -> dict:
        """
        Train the MLP on a complete dataset using tolerance-based early stopping.
        Computes training loss and validation loss at each epoch.

        Args:
            train_data: DataFrame with 100 feature columns and a 'class' column.
            val_data: Validation DataFrame (same format as train_data).
            tolerance: Minimum difference between losses to continue training.
            max_epochs: Maximum number of epochs to train.
            verbose: Whether to print training progress.
        Returns:
            Dictionary with training and validation history.
        """
        train_losses = []
        val_losses = []
        prev_avg_loss = float("inf")
        epoch = 0

        if verbose:
            print(" " + "_" * 41 + " ")
            print(f"| {'Epoch':^5} | {'Training Loss':^15} | {'Validation Loss':^15} |")

        while epoch < max_epochs:
            epoch_loss = 0

            # Training
            for i in range(len(train_data)):
                sample = np.array(train_data.iloc[i, :100].values, dtype=np.float64)
                d = EXPECTED_OUTPUT[train_data.iloc[i, 100]]

                activations = self.feedforward(sample)
                y_pred = activations[-1]
                loss = self.calculate_loss(d, y_pred)
                epoch_loss += loss

                self.params = self.backpropagation(sample, d, activations)

            avg_loss = epoch_loss / len(train_data)
            train_losses.append(avg_loss)

            # Validation metrics
            val_epoch_loss = 0
            for i in range(len(val_data)):
                val_sample = np.array(val_data.iloc[i, :100].values, dtype=np.float64)
                val_d = EXPECTED_OUTPUT[val_data.iloc[i, 100]]

                val_y_pred = self.feedforward(val_sample)[-1]
                val_loss = self.calculate_loss(val_d, val_y_pred)
                val_epoch_loss += val_loss

            val_avg_loss = val_epoch_loss / len(val_data)
            val_losses.append(val_avg_loss)

            # Print progress
            if verbose and epoch % 25 == 0:
                print(f"| {epoch:5} |  {avg_loss:13.10f} |  {val_avg_loss:13.12f} |")

            # Check for stop condition
            loss_change = abs(avg_loss - prev_avg_loss)
            if loss_change < tolerance:
                if verbose:
                    print(" " + "¯" * 41 + " ")
                    print(
                        f"Stopped at epoch {epoch}: Loss change {loss_change:.2e} < tolerance={tolerance}"
                    )
                break

            prev_avg_loss = avg_loss
            epoch += 1

        self.history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epochs": epoch,
        }

        return self.history

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_data: DataFrame with 100 feature columns and a 'class' column.
        Returns:
            Dictionary with evaluation metrics.
        """
        correct = 0
        total = len(test_data)

        for i in range(total):
            sample = np.array(test_data.iloc[i, :100].values, dtype=np.float64)
            true_class = test_data.iloc[i, 100]
            predicted = self.classify(sample)
            if predicted == true_class:
                correct += 1

        accuracy = correct / total
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    def save(self, filepath: Path) -> None:
        """
        Save the model to disk with all parameters and metadata.

        Args:
            filepath: Path where to save the model (should end with .pkl)
        """
        model_data = {
            "params": self.params,
            "hyperparameters": {
                "activation_type": self.activation_type,
                "learning_rate": self.learning_rate,
                "momentum": self.momentum,
                "seed": self.seed,
            },
            "metadata": self.metadata,
            "history": self.history,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "MLP":
        """
        Load a model from disk.

        Args:
            filepath: Path to the saved model file.
        Returns:
            Loaded MLP instance.
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Recreate model with saved hyperparameters
        hyperparams = model_data["hyperparameters"]
        model = cls(
            activation_type=hyperparams["activation_type"],
            learning_rate=hyperparams["learning_rate"],
            momentum=hyperparams["momentum"],
            seed=hyperparams["seed"],
        )

        # Restore parameters and metadata
        model.params = model_data["params"]
        model.metadata = model_data["metadata"]
        model.history = model_data["history"]

        return model

    def get_info(self) -> Dict:
        """Get comprehensive model information."""
        return {
            "metadata": self.metadata,
            "hyperparameters": {
                "activation_type": self.activation_type,
                "learning_rate": self.learning_rate,
                "momentum": self.momentum,
                "seed": self.seed,
            },
            "training_history": {
                "epochs": self.history.get("epochs", 0),
                "final_train_loss": self.history.get("train_losses", [None])[-1],
                "final_val_loss": self.history.get("val_losses", [None])[-1],
            },
        }
