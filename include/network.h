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
        void computeErrors(const vector<double>& expectedOutput);
        void updateWeights(double learningRate, double momentumFactor);
        void train(const vector<vector<double>>& X, 
                   const vector<vector<double>>& Y, 
                   double learningRate, 
                   double momentumFactor, 
                   int maxEpochs, 
                   double minError);
};

vector<vector<double>> oneHotEncode(const vector<double>& labels, int numClasses);
vector<vector<double>> normalize(const vector<vector<double>>& X, vector<double>& minVals, vector<double>& maxVals);

#endif