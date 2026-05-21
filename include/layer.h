#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "neuron.h"

using namespace std;

enum LayerType{
    INPUT,
    HIDDEN,
    OUTPUT
};
enum NetType{
    REGRESSION,
    CLASSIFICATION
};

class Layer{
    public:
        vector<Neuron> neuronsLayer;
        vector<double> output;
        int numNeurons;
        LayerType type;

        Layer(int numNeurons, int numInputsPerNeuron, LayerType type, mt19937& generator);
        void forward(vector<double>& inputs, NetType type);
};

#endif