#include "network.h"

Network::Network(vector<int> topology, mt19937& generator, NetType type) {
    int numLayers = topology.size();
    this->type = type;
    this->layers.push_back(Layer(topology[0], 0, INPUT, generator));

    for (int i = 1; i < numLayers - 1; i++) {
        int numNeurons = topology[i];
        int numInputs = topology[i - 1]; 
        
        this->layers.push_back(Layer(numNeurons, numInputs, HIDDEN, generator));
    }

    int outputIndex = numLayers - 1;
    int numOutputNeurons = topology[outputIndex];
    int numOutputInputs = topology[outputIndex - 1];

    this->layers.push_back(Layer(numOutputNeurons, numOutputInputs, OUTPUT, generator));
}

void Network::predict(const vector<double>& sample) {
    this->layers[0].forward(sample, this->type);

    for (int i = 1; i < this->layers.size(); i++) {
        this->layers[i].forward(this->layers[i - 1].output, this->type);
    }

    this->output = this->layers.back().output;
}