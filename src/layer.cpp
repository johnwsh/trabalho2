#include "../include/layer.h"

Layer::Layer(int numNeurons, int numInputsPerNeuron, LayerType type, mt19937& generator) {
    this->numNeurons = numNeurons;
    this->type = type;
    
    for (int i = 0; i < numNeurons; i++) {
        this->neuronsLayer.push_back(Neuron(numInputsPerNeuron, generator));
    }
}

void Layer::forward(vector<double>& inputs, NetType type) {
    this->output.clear(); 

    if (this->type == INPUT) {
        this->output = inputs;
        
        for (int i = 0; i < this->neuronsLayer.size(); i++) {
            this->neuronsLayer[i].output = inputs[i];
        }
    } 
    else if (type == REGRESSION and this->type == OUTPUT){
        for (int i = 0; i < this->neuronsLayer.size(); i++) {
            
            this->neuronsLayer[i].activate(inputs, false); 
            this->output.push_back(neuronsLayer[i].output);
        }
    }
    else {
        for (int i = 0; i < this->neuronsLayer.size(); i++) {
            
            this->neuronsLayer[i].activate(inputs, true); 
            this->output.push_back(neuronsLayer[i].output);
        }
    }
}