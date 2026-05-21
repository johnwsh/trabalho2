#include "../include/utils.h"

vector<double> parseCSVRow(string& row) {
    vector<double> fields;
    stringstream ss(row);
    string field;

    while (getline(ss, field, ',')) {
        try {
            fields.push_back(stod(field));
        }
        catch (...) {
            fields.push_back(0.0);
        }
    }

    return fields;
}

vector<vector<double>> readCSV(string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return data;
    }

    string line;

    while (getline(file, line)) {
        if(line.empty()) continue; 
        data.push_back(parseCSVRow(line));
    }

    return data; 
}

//Para separar em dados e prediction
void splitDataset(  vector<vector<double>>& dataset, 
                    int gravityIndex,
                    int classIndex, 
                    vector<vector<double>>& X, 
                    vector<double>& Y,
                    vector<int>& Z) {
    
    for (auto& row : dataset) {
        vector<double> features;
        
        for (int i = 3; i < row.size(); i++) {
            if (i == gravityIndex) {
                Y.push_back(row[i]);
            } else if (i == classIndex) {
                Z.push_back(row[i]);
            } else {
                features.push_back(row[i]);
            }
        }
        X.push_back(features);
    }
}

double calculateRMSE(vector<double>& real, vector<double>& prediction) {
    double mse = 0;
    for (int i = 0; i < real.size(); i++){
        double erro = real[i] - prediction[i];
        mse += erro * erro;
    }

    return sqrt(mse / real.size());
}

double calculateMAE(vector<double>& real, vector<double>& prediction) {
    double mae = 0;
    for (int i = 0; i < real.size(); i++){
        mae += abs(real[i] - prediction[i]);
    }

    return mae / real.size();
}

double calculateR2(vector<double>& real, vector<double>& prediction) {
    double mean = 0;
    for (double v : real)
        mean += v;

    mean /= real.size();

    double ss_res = 0;
    double ss_tot = 0;

    for (int i = 0; i < real.size(); i++){
        ss_res += pow(real[i] - prediction[i], 2);
        ss_tot += pow(real[i] - mean, 2);
    }

    return 1.0 - (ss_res / ss_tot);
}

double calculatePrecision(int TP, int FP) {
    return (double)TP / (TP + FP);
}

double calculateRecall(int TP, int FN) {
    return (double)TP / (TP + FN);
}

double calculateF1Score(double precision, double recall) {
    return 2.0 * ((precision * recall) / (precision + recall));
}

double calculateAccuracy(vector<int>& real, vector<int>& prediction) {
    int correct = 0;
    for (int i = 0; i < real.size(); i++){
        if (real[i] == prediction[i])
            correct++;
    }

    return (double)correct / real.size();
}
void confusionMatrix(vector<int>& real, vector<int>& prediction, vector<vector<int>>& matrix) {
    for (int i = 0; i < real.size(); i++){
        int realClass = (int)real[i] - 1;
        int predClass = (int)prediction[i] - 1;
        if (
            realClass >= 0 &&
            realClass < 4 &&
            predClass >= 0 &&
            predClass < 4
        )
        {
            matrix[realClass][predClass]++;
        }
    }
}