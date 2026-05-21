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

void Network::computeErrors(const vector<double>& expectedOutput) {

    Layer& outputLayer = this->layers.back(); 
    
    for (int i = 0; i < outputLayer.neuronsLayer.size(); i++) {
        //erro bruto do output
        Neuron& n = outputLayer.neuronsLayer[i];
        
        double error = expectedOutput[i] - n.output;
        
        if (this->type == REGRESSION) {
            n.delta = error; 
        } else {
            //derivada sigmoid
            n.delta = error * (n.output * (1.0 - n.output));
        }
    }

    for (int l = this->layers.size() - 2; l > 0; l--) {
        //erro das camadas ocultas
        Layer& currentLayer = this->layers[l];
        Layer& nextLayer = this->layers[l + 1]; 
        
        for (int i = 0; i < currentLayer.neuronsLayer.size(); i++) {
            Neuron& n = currentLayer.neuronsLayer[i];
            double somaDosErros = 0.0;
            
            for (int j = 0; j < nextLayer.neuronsLayer.size(); j++) {
                Neuron& nextNode = nextLayer.neuronsLayer[j];
                
                somaDosErros += nextNode.weights[i] * nextNode.delta;
            }
            //derivada sigmoid
            n.delta = somaDosErros * (n.output * (1.0 - n.output));
        }
    }
}