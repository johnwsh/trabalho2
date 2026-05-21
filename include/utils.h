#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <cmath>
#include <map>
#include <algorithm>

using namespace std;

vector<double> parseCSVRow(string& row);
vector<vector<double>> readCSV(string& filename);
void splitDataset(  vector<vector<double>>& dataset, 
                    int gravityIndex,
                    int classIndex, 
                    vector<vector<double>>& X, 
                    vector<double>& Y,
                    vector<int>& Z);

double calculateRMSE(vector<double>& real, vector<double>& prediction); //Raiz do erro médio quadrático
double calculateMAE(vector<double>& real, vector<double>& prediction); //Erro médio absoluto
double calculateR2(vector<double>& real, vector<double>& prediction); //Coeficiente de determinação
double calculatePrecision(int TP, int FP);
double calculateRecall(int TP, int FN);
double calculateF1Score(double precision, double recall);
double calculateAccuracy(vector<int>& real, vector<int>& prediction); //Acurácia para classificação
void confusionMatrix(vector<int>& real, vector<int>& prediction, vector<vector<int>>& matrix); //Matriz de confusão para classificação

#endif 