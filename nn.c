#include <stdio.h>
#include "nn.h"

// Initialize weights with arbitrary initial values
void initWeights(Perceptron *n) {
    for (short i = 0; i < INPUT_SIZE; i++) {
        n->weights[i] = 0.4f;
    }
}

// Compute weighted sum and apply binary step activation function
void activationFunction(Perceptron *n) {
    float sum = 0.0f;
    for (short i = 0; i < INPUT_SIZE; i++) {
        sum += (float)n->inputs[i] * n->weights[i];
    }

    // Step activation threshold
    if (sum >= 0.5f) {
        n->output = 1.0f;
    } else {
        n->output = 0.0f;
    }
}

// Update weights using Perceptron Learning Rule
void updateWeights(Perceptron *n, float target) {
    float error = target - n->output;
    for (short i = 0; i < INPUT_SIZE; i++) {
        n->weights[i] += LEARNING_RATE * error * (float)n->inputs[i];
    }
}

// Train perceptron across all epochs and target samples
void train(Perceptron *n, int epochs, int inputs[][INPUT_SIZE], int targets[]) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int j = 0; j < TARGETS; j++) {
            // Copy row array elements into perceptron inputs
            for (short k = 0; k < INPUT_SIZE; k++) {
                n->inputs[k] = inputs[j][k];
            }

            activationFunction(n);
            updateWeights(n, (float)targets[j]);
        }
    }
}

int main() {
    Perceptron p;
    initWeights(&p);

    // Sample dataset
    int inputs[TARGETS][INPUT_SIZE] = {
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1},
        {1, 0, 1},
        {1, 1, 0},
        {0, 1, 1}
    };

    int targets[TARGETS] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 0};

    printf("--- Training Perceptron ---\n");
    train(&p, 100, inputs, targets);

    printf("\n--- Test Evaluation ---\n");
    for (int i = 0; i < TARGETS; i++) {
        for (short k = 0; k < INPUT_SIZE; k++) {
            p.inputs[k] = inputs[i][k];
        }
        activationFunction(&p);
        printf("[%d, %d, %d] -> Prediction: %.0f | Target: %d\n",
               p.inputs[0], p.inputs[1], p.inputs[2], p.output, targets[i]);
    }

    printf("\nFinal Weights: ");
    for (int i = 0; i < INPUT_SIZE; i++) {
        printf("w%d = %.4f ", i, p.weights[i]);
    }
    printf("\n");

    return 0;
}
