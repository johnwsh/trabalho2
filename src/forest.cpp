#include "../include/forest.h"

Forest::Forest(int numTrees, TreeType type, vector<vector<double>> &X, vector<double> &Y, mt19937 &generator, int treeDepth){
    this->type = type;
    for (int i = 0; i < numTrees; i++){
        
        vector<vector<double>> Xb;
        vector<double> Yb;

        buildBootstrappedDataset(X,Y,X.size(),generator,Xb,Yb);
        Tree* tree = buildTree(treeDepth, type, Xb, Yb, generator);
        forest.push_back(tree);
    }
}

#include <map>

double Forest::predict(vector<double>& sample){
    if (this->type == REGRESSION) {
        double sumPredict = 0;
        for (int i = 0; i < forest.size(); i++){
            sumPredict += forest[i]->predict(sample);
        }
        return sumPredict / forest.size();
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