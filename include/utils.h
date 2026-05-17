#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

vector<double> parseCSVRow(const string& row);
vector<vector<double>> readCSV(const string& filename);
void splitDataset(const vector<vector<double>>& dataset, int predictionIndex, vector<vector<double>>& X, vector<double>& y);

#endif 