/*
Define the structure of a perceptron and the operations
applicable to it
*/

#ifndef NN_H
#define NN_H

#define INPUT_SIZE 3
#define LEARNING_RATE 0.03
#define TARGETS 10

typedef struct{
    int inputs[INPUT_SIZE];
    float weights[INPUT_SIZE];
    float output;
}Perceptron;

void initWeights(Perceptron *n);
void activationFunction(Perceptron *n);
void updateWeights(Perceptron *n, float target);
void train(Perceptron *n, int epochs, int inputs[][INPUT_SIZE], int targets[]);

#endif // NN_H
