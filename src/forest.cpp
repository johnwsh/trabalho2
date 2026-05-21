#include "../include/forest.h"

Forest::Forest(int numTrees, TreeType type, vector<vector<double>> &X, vector<double> &Y, mt19937 &generator, int treeDepth){
    this->type = type;
    for (int i = 0; i < numTrees; i++){
        
        vector<vector<double>> Xb;
        vector<double> Yb;

        vector<bool> usedSamples(X.size(), false);

        buildBootstrappedDataset(X,Y,X.size(),generator,Xb,Yb, usedSamples);
        Tree* tree = buildTree(treeDepth, type, Xb, Yb, usedSamples, generator);
        forest.push_back(tree);
    }
}

#include <map>

double Forest::predict(vector<double>& sample, int sampleIndex){
    double sumPredict = 0;
    int usedTrees = 0;
    for (int i = 0; i < forest.size(); i++){
        if(forest[i]->usedSamples[sampleIndex]) {
            sumPredict += forest[i]->predict(sample);
            usedTrees++;
        }
    }
    return sumPredict / usedTrees; 
}