#include "neuron.h"
#include <math.h>

Neuron::Neuron(int numInputs, mt19937& generator) {
    //constroi cada neuronio com pesos aleatorios
    this->numInputs = numInputs;
    
    uniform_real_distribution<double> dist(-1.0, 1.0);
    
    this->bias = dist(generator); 
    
    for (int i = 0; i < numInputs; i++) {
        this->weights.push_back(dist(generator));
    }
}

void Neuron::activate(const vector<double> &features, bool useActivationFunction){
    double u = this->bias; 
    
    for (int i = 0; i < features.size(); i++) {
        u += features[i] * this->weights[i];
    }
    if (useActivationFunction)
        this->output = 1.0 / (1.0 + exp(-u)); 
    else 
        this->output = u;
}
