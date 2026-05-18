#include "../include/trees.h"
#include <algorithm>
#include <limits.h>
#include <numeric>

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
        if (attAndY[i].first == attAndY[i+1].first) continue;
        double splitCandidate = (attAndY[i].first + attAndY[i+1].first)/2;
        
        sumLeft += attAndY[i].second;
        sumRight -= attAndY[i].second;

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
           vector<vector<double>>& X, 
           vector<double>& y, 
           mt19937 &generator,
           int currentDepth = 0){

    Node* node = new Node();
    bool isEqual = true;
    for(int i = 0; i < y.size()-1; i++){
        if (y[i] != y[i+1]){
            isEqual = false;
            break;
        }
    }

    if (currentDepth >= maxDepth || X.size() < minSamplesSplit || isEqual){
        node->isLeaf = true;
        node->value = accumulate(y.begin(),y.end(), 0.0)/y.size();
        return node;
    }

    uniform_int_distribution<int> dist(0,X[0].size()-1);
    int attToRemove = dist(generator);

    double threshold;
    double thresholdLocal;
    int attIndex;
    double error = INT_MAX;

    for (int i = 0; i < X[0].size(); i ++){
        if (i != attToRemove){
            double localError = calculateMSE(i, X, y, thresholdLocal, generator);
            if (localError < error){
                threshold = thresholdLocal;
                error = localError;
                attIndex = i;
            }
        }
    }

    node->featureIndex = attIndex;
    node->threshold = threshold;
    node->isLeaf = false;

    vector<vector<double>> X_left, X_right;
    vector<double> y_left, y_right;

    for (int i = 0; i < X.size(); i++) {
        if (X[i][attIndex] <= threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }

    node->left = CART(maxDepth, minSamplesSplit, X_left, y_left, generator, currentDepth + 1);
    
    node->right = CART(maxDepth, minSamplesSplit, X_right, y_right, generator, currentDepth + 1);

    return node;
}

//Utilizado para abstração
Tree* buildTree(int maxDepth, TreeType type, 
                vector<vector<double>>& X, 
                vector<double>& y,
                mt19937 &generator){
    
    Tree* tree = new Tree(type);
    int minSamplesSplit;

    if (type == CLASSIFICATION)
        minSamplesSplit = 1;
    else 
        minSamplesSplit = 5;
    tree->root = CART (maxDepth, minSamplesSplit, X, y, generator);
    return tree;
}