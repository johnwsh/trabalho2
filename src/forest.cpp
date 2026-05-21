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
    if (this->type == REGRESSION) {
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
    else { 

        vector<double> predictions;
        for (int i = 0; i < forest.size(); i++) {
            predictions.push_back(forest[i]->predict(sample));
        }

        double winningClass = predictions[0];
        int maxVotos = 0;
        
        for (int i = 0; i < predictions.size(); i++) {
            int votos = 0;
            for (int j = 0; j < predictions.size(); j++) {
                if (predictions[j] == predictions[i]) votos++;
            }
            if (votos > maxVotos) {
                maxVotos = votos;
                winningClass = predictions[i];
            }
        }
        
        return winningClass;
    }
}