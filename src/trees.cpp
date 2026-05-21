#include "../include/trees.h"
#include <algorithm>
#include <limits.h>
#include <numeric>
#include <map>

void buildBootstrappedDataset(  vector<vector<double>>& X, 
                                vector<double>& Y, 
                                int numSamples, 
                                mt19937 &generator, 
                                vector<vector<double>>& newDatasetX, 
                                vector<double>& newDatasetY,
                                vector<bool>& usedSamples){

    uniform_int_distribution<int> dist(0, X.size() - 1);

    for (int i = 0; i < numSamples; i++){
        int datasetIndex = dist(generator);
        newDatasetX.push_back(X[datasetIndex]);
        newDatasetY.push_back(Y[datasetIndex]);
        usedSamples[datasetIndex] = true;
    }

}

double calculateGini(int attIndex, vector<vector<double>> &X, vector<double> &Y, double &threshold, mt19937 &generator){
    vector<pair<double, double>> attAndY(X.size());
    
    map<double, int> rightCounts; 
    map<double, int> leftCounts;

    for (int i = 0; i < X.size(); i++) {
        attAndY[i].first = X[i][attIndex];
        attAndY[i].second = Y[i];
        
        //mesma otimização que MSE, só que dessa vez conta quantos aparecem por classe
        rightCounts[Y[i]]++; 
    }
    sort(attAndY.begin(), attAndY.end());

    double gini = INT_MAX;
    int countRight = X.size();
    int countLeft = 0;

    for (int i = 0; i < attAndY.size() -1; i++){

        rightCounts[attAndY[i].second]--;
        leftCounts[attAndY[i].second]++;
        countLeft++;
        countRight--;

        if (attAndY[i].first == attAndY[i + 1].first) {
            continue;
        }

        double giniLeft = 1.0;
        map<double, int>::iterator it;
        for (it = leftCounts.begin(); it != leftCounts.end(); it++) {
            int contagem = it->second;
            if (contagem > 0) {
                double p = (double)contagem / countLeft;
                giniLeft -= (p * p);
            }
        }

        double giniRight = 1.0;
        for (it = rightCounts.begin(); it != rightCounts.end(); it++) {
            int contagem = it->second;
            if (contagem > 0) {
                double p = (double)contagem / countRight;
                giniRight -= (p * p);
            }
        }

        double weightedGini = ((countLeft * giniLeft) + (countRight * giniRight)) / attAndY.size();
        if (weightedGini < gini) {
            gini = weightedGini;
            threshold = (attAndY[i].first + attAndY[i + 1].first) / 2.0;
        }
    }

    return gini;
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

        sumLeft += attAndY[i].second;
        sumRight -= attAndY[i].second;

        if (attAndY[i].first == attAndY[i+1].first) continue;
        double splitCandidate = (attAndY[i].first + attAndY[i+1].first)/2;

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
           TreeType treetype,
           int currentDepth){

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
        if (treetype == REGRESSION) {
            node->value = accumulate(y.begin(), y.end(), 0.0) / y.size();
        } 
        else { 
            double winningClass = y[0];
            int maxVotos = 0;
            
            for (int i = 0; i < y.size(); i++) {
                int votos = 0;
                for (int j = 0; j < y.size(); j++) {
                    if (y[j] == y[i]) votos++;
                }
                if (votos > maxVotos) {
                    maxVotos = votos;
                    winningClass = y[i];
                }
            }
            node->value = winningClass;
        }
        return node;
    }

    uniform_int_distribution<int> dist(0,X[0].size()-1);
    int attToRemove = dist(generator);

    double threshold;
    double thresholdLocal;
    int attIndex = -1;
    double error = INT_MAX;

    for (int i = 0; i < X[0].size(); i ++){
        if (i != attToRemove){
            double localError;
            
            if (treetype == REGRESSION)
                localError = calculateMSE(i, X, y, thresholdLocal, generator);
            else 
                localError = calculateGini(i, X, y, thresholdLocal, generator);

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

    node->left = CART(maxDepth, minSamplesSplit, X_left, y_left, generator, treetype, currentDepth + 1);
    
    node->right = CART(maxDepth, minSamplesSplit, X_right, y_right, generator, treetype, currentDepth + 1);

    return node;
}

//Utilizado para abstração
Tree* buildTree(int maxDepth, TreeType type, 
                vector<vector<double>>& X, 
                vector<double>& y,
                vector<bool>& usedSamples,
                mt19937 &generator){ 
    
    Tree* tree = new Tree(type);
    tree->usedSamples = usedSamples;

    int minSamplesSplit;

    if (type == CLASSIFICATION)
        minSamplesSplit = 1;
    else 
        minSamplesSplit = 5;
    tree->root = CART (maxDepth, minSamplesSplit, X, y, generator, type);
    return tree;
}

double Tree::predict(vector<double>& sample){
    Node* travellingNode = root;
    while(!travellingNode->isLeaf){
        if (sample[travellingNode->featureIndex] <= travellingNode->threshold){
            travellingNode = travellingNode->left;
        }
        else {
            travellingNode = travellingNode->right;
        }
    }
    return travellingNode->value;
}