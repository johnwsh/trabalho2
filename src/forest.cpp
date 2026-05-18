#include "../include/forest.h"

Forest::Forest(int numTrees, TreeType type, vector<vector<double>> X, vector<double> Y, mt19937 &generator, int treeDepth){
    type = type;
    for (int i = 0; i < numTrees; i++){
        
        vector<vector<double>> Xb;
        vector<double> Yb;

        buildBootstrappedDataset(X,Y,X.size(),generator,Xb,Yb);
        Tree* tree = buildTree(treeDepth, type, Xb, Yb, generator);
        forest.push_back(tree);
    }
}

double Forest::predict(vector<double>& sample){
    double sumPredict = 0;

    for (int i = 0; i < forest.size(); i++){
        sumPredict += forest[i]->predict(sample);
    }
    return sumPredict/forest.size();
}