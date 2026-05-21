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
        void predict(vector<double>& sample);
        void computeErrors(vector<double>& expectedOutput);
        void updateWeights(double learningRate, double momentumFactor);
        void train(vector<vector<double>>& X, 
                   vector<vector<double>>& Y, 
                   double learningRate, 
                   double momentumFactor, 
                   int maxEpochs, 
                   double minError);
};

vector<vector<double>> oneHotEncode(vector<double>& labels, int numClasses);
vector<vector<double>> normalize(vector<vector<double>>& X, vector<double>& minVals, vector<double>& maxVals);
vector<vector<double>> applyNormalization(const vector<vector<double>>& X_test, const vector<double>& minVals, const vector<double>& maxVals);

#endif