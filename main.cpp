#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>

#include "include/utils.h"
#include "include/trees.h"
#include "include/forest.h"

using namespace std;

#define NUM_TREES 100
#define TREE_DEPTH 5
#define TRAIN_PERCENT 0.8

int main(){
    //ler o dataset e armazenar na memória
    mt19937 generator(time(0));
    vector<vector<double>> dataset = readCSV("01_treino_sinais_vitais_sem_label.txt");
    vector<vector<double>> X;
    vector<double> Y;

    splitDataset(dataset, 6, X, Y);

    //separar o dataset em treino e teste (80% treino, 20% teste)
    vector<int> indices(X.size());

    for (int i = 0; i < indices.size(); i++){
        indices[i] = i;
    }

    shuffle(indices.begin(), indices.end(), generator);

    int trainSize = X.size() * TRAIN_PERCENT;

    vector<vector<double>> X_train;
    vector<double> Y_train;

    vector<vector<double>> X_test;
    vector<double> Y_test;

    for (int i = 0; i < indices.size(); i++){

        int idx = indices[i];

        if (i < trainSize){
            X_train.push_back(X[idx]);
            Y_train.push_back(Y[idx]);
        }
        else{
            X_test.push_back(X[idx]);
            Y_test.push_back(Y[idx]);
        }
    }
    
    Forest forestRegres(NUM_TREES, REGRESSION, X_train, Y_train, generator, TREE_DEPTH);
    vector<double> OOB_gravity;
    for (int i = 0; i < X_train.size(); i++){
        double prediction = forestRegres.predict(X_train[i], i);
        OOB_gravity.push_back(prediction);
    }
    Forest forestClass(NUM_TREES, CLASSIFICATION, X_train, OOB_gravity, generator, TREE_DEPTH);
    vector<double> OOB_class;
    for (int i = 0; i < X_train.size(); i++){
        double prediction = forestClass.predict(X_train[i], i);
        OOB_class.push_back(prediction);
    }

    ofstream outFile("gravidade.txt");
    outFile << "real,predito\n";

    for (int i = 0; i < Y_train.size(); i++){
        outFile
            << Y_train[i]
            << ","
            << OOB_gravity[i]
            << "\n";
    }
    outFile.close();

    ofstream outClass("classe.txt");
    outClass << "real\n";

    for (int i = 0; i < OOB_class.size(); i++){
        outClass
            << OOB_class[i]
            << "\n";
}

outClass.close();

    return 0;
}