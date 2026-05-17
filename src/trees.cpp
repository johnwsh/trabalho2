#include "../include/trees.h"

void buildBootstrappedDataset(vector<vector<double>>& X, 
                                                vector<double>& Y, 
                                                int numSamples, 
                                                mt19937 &generator, 
                                                vector<vector<double>>& newDatasetX, 
                                                vector<double>& newDatasetY){

    uniform_int_distribution<int> dist(1, X.size() - 1);

    for (int i = 0; i < numSamples; i++){
        int datasetIndex = dist(generator);
        newDatasetX.push_back(X[datasetIndex]);
        newDatasetY.push_back(Y[datasetIndex]);
    }

}

double calculateMSE(int attIndex, vector<vector<double>> &X, vector<double> &Y, double &threshold){
    
}

Node* CART(int maxDepth, 
           int minSamplesSplit,
           const vector<vector<double>>& X, 
           const vector<double>& y, 
           int currentDepth = 0, 
           mt19937 &generator){

    uniform_int_distribution<int> dist(0,X[0].size()-1);
    int attToRemove = dist(generator);

    for (int i = 0; i < X[0].size(); i ++){
        if (i != attToRemove){
            
        }
    }
}

//Utilizado para abstração
Tree* buildTree(int maxDepth, int minSamplesSplit, TreeType type, 
                const vector<vector<double>>& X, 
                const vector<double>& y,
                mt19937 &generator){
    
    Tree* tree = new Tree(type);
    tree->root = CART (maxDepth, minSamplesSplit, X, y, 0, generator);
    return tree;
}