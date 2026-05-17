#include <iostream>
#include <vector>
#include <string>

#include "include/utils.h"

using namespace std;

int main(){
    //ler o dataset e armazenar na memória
    vector<vector<double>> dataset = readCSV("01_treino_sinais_vitais_sem_label.txt");

    cout << "Dataset" << endl;

    for (int i = 0; i < dataset.size(); i++){
        for (int j = 0; j < dataset[i].size(); j++){
            cout << dataset[i][j] << ", ";
        }
        cout << "\n";
    }


    return 0;
}

