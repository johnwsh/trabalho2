#include "../include/utils.h"

using namespace std;

vector<double> parseCSVRow(const string& row) {
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

vector<vector<double>> readCSV(const string& filename) {
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

#include <vector>
#include <iostream>

//Para separar em dados e prediction
void splitDataset(const vector<vector<double>>& dataset, int predictionIndex, vector<vector<double>>& X, vector<double>& y) {
    
    for (const auto& row : dataset) {
        vector<double> features;
        
        for (int i = 3; i < row.size(); i++) {
            if (i == predictionIndex) {
                y.push_back(row[i]);
            } else {
                features.push_back(row[i]);
            }
        }
        X.push_back(features);
    }
}