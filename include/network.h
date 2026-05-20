#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "layer.h"

using namespace std;

class Network{
    public:
        vector<Layer> layers;
        vector<double> output;
        NetType type;

        Network(vector<int> topology, mt19937& generator, NetType type); 
        void predict(const vector<double>& sample);

};

#endif