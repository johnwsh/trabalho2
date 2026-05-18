#include <iostream>
#include <vector>
#include <string>
#include <random>

#include "include/utils.h"
#include "include/trees.h"
#include "include/forest.h"

using namespace std;

int main(){
    //ler o dataset e armazenar na memória
    mt19937 generator(time(0));
    vector<vector<double>> dataset = readCSV("01_treino_sinais_vitais_sem_label.txt");
    vector<vector<double>> X;
    vector<double> Y;

    splitDataset(dataset, 6, X, Y);

    vector<vector<double>> Xb;
    vector<double> Yb;

    buildBootstrappedDataset(X, Y, X.size(), generator, Xb, Yb);

    return 0;
}

