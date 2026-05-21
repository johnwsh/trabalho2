#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>

using namespace std;

class Neuron{
    public:
        double bias;
        vector<double> weights;
        vector<double> momentum;
        double output;
        int numInputs;
        double delta;
        double biasMomentum;
    
        void activate(const vector<double> &features, bool useActivationFunction);
        Neuron(int numInput, mt19937& generator);
        void updateWeights(const vector<double>& inputs, double learningRate, double momentumFactor);
};

#endif