#include "../include/trees.h"
#include <algorithm>
#include <limits.h>

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

double calculateMSE(int attIndex, vector<vector<double>> &X, vector<double> &Y, double &threshold, mt19937 &generator){

    vector<pair<double, double>> attAndY(X.size());
    double sumTotal = 0;
    for (int i = 0; i < X.size(); i++) {
        attAndY[i].first = X[i][attIndex];
        attAndY[i].second = Y[i];
        //otimização espertinha pra não precisar recalcular
        sumTotal += Y[i];
    }
    sort(attAndY.begin(), attAndY.end());

    double error = INT_MAX;
    double sumLeft = 0;
    double sumRight = sumTotal;

    for (int i = 0; i < attAndY.size() -1; i++){
        double splitCandidate = (attAndY[i].first + attAndY[i+1].first)/2;
        
        sumLeft += attAndY[i].first;
        sumRight -= attAndY[i].first;

        double LeftMean = sumLeft/(i+1);
        double RightMean = sumRight/(attAndY.size() - (i + 1));

        double localError = 0;
        for (int j = 0; j < attAndY.size(); j++){
            if (attAndY[j].first < splitCandidate){
                localError += (attAndY[j].second - LeftMean)*(attAndY[j].second - LeftMean);
            }
            else {
                localError += (attAndY[j].second - RightMean)*(attAndY[j].second - RightMean);
            }
        }

        if (localError < error){
            error = localError;
            threshold = splitCandidate;
        }
    }
    return error;
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