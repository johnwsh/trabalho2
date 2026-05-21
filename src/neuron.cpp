#include "../include/neuron.h"
#include <math.h>

Neuron::Neuron(int numInputs, mt19937& generator) {
    this->numInputs = numInputs;
    uniform_real_distribution<double> dist(-1.0, 1.0);
    
    this->bias = dist(generator); 
    this->biasMomentum = 0.0; 
    
    for (int i = 0; i < numInputs; i++) {
        this->weights.push_back(dist(generator));
        this->momentum.push_back(0.0); 
    }
}

void Neuron::activate(vector<double> &features, bool useActivationFunction){
    double u = this->bias; 
    
    for (int i = 0; i < features.size(); i++) {
        u += features[i] * this->weights[i];
    }
    if (useActivationFunction)
        this->output = 1.0 / (1.0 + exp(-u)); 
    else 
        this->output = u;
}

void Neuron::updateWeights(vector<double>& inputs, double learningRate, double momentumFactor) {
    
    for (int i = 0; i < this->weights.size(); i++) {
        this->momentum[i] = (momentumFactor * this->momentum[i]) + (learningRate * this->delta * inputs[i]);
        
        this->weights[i] += this->momentum[i];
    }
    
    this->biasMomentum = (momentumFactor * this->biasMomentum) + (learningRate * this->delta);
    this->bias += this->biasMomentum;
}
