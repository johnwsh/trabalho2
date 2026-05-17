#include "../include/utils.h"

using namespace std;

vector<double> parseCSVRow(const string& row) {
    vector<double> fields;
    size_t start = 0;
    bool inQuotes = false;
    
    for (size_t i = 0; i < row.length(); ++i) {
        if (!inQuotes && row[i] == ',') {
            string fieldStr = row.substr(start, i - start);
            try {
                fields.push_back(stod(fieldStr)); 
            } catch (const invalid_argument& e) {
                fields.push_back(0.0); 
            }
            start = i + 1;
        } else if (row[i] == '"') {
            inQuotes = !inQuotes;
        }
    }
    
    string lastField = row.substr(start);
    try {
        if (!lastField.empty()) {
            fields.push_back(stod(lastField));
        }
    } catch (const invalid_argument& e) {
        fields.push_back(0.0);
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
        
        for (int i = 0; i < row.size(); i++) {
            if (i == predictionIndex) {
                y.push_back(row[i]);
            } else {
                features.push_back(row[i]);
            }
        }
        X.push_back(features);
    }
}