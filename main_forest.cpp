#include <iostream>
#include <vector>
#include <string>
#include <random>  
#include <algorithm>
#include <fstream>
#include <time.h>
#include <cmath> 
#include <map>

#include "include/utils.h"
#include "include/forest.h" 
#include "include/trees.h"

using namespace std;

// ==========================================
// PAINEL DE CONTROLE DA RANDOM FOREST
// ==========================================
#define K_FOLDS 5
#define NUM_TREES 100
#define TREE_DEPTH 10
// ==========================================

int main(){
    mt19937 generator(time(0));
    string filename = "02_treino_sinais_vitais_com_label.txt";
    
    vector<vector<double>> dataset = readCSV(filename);
    vector<vector<double>> X; // Features
    vector<double> Y;         // Gravidade (Regressão)
    vector<int> Z;            // Classes (Classificação)

    splitDataset(dataset, 6, 7, X, Y, Z);

    // O construtor da Forest exige vector<double>, então convertemos o Z
    vector<double> Z_double(Z.begin(), Z.end());

    // Embaralha o dataset completo
    vector<int> indices(X.size());
    for (int i = 0; i < indices.size(); i++){
        indices[i] = i;
    }
    shuffle(indices.begin(), indices.end(), generator);

    cout << "========== INICIANDO " << K_FOLDS << "-FOLD CROSS VALIDATION ==========" << endl;
    cout << "Modelos: 1 Floresta de Regressao + 1 Floresta de Classificacao" << endl;
    cout << "Dataset total: " << X.size() << " pacientes." << endl;
    cout << "Hiperparametros -> Arvores: " << NUM_TREES << " | Profundidade Max: " << TREE_DEPTH << endl;
    cout << "=================================================================" << endl;
    
    int foldSize = X.size() / K_FOLDS;
    
    // Acumuladores de métricas
    double soma_rmse_regressao = 0.0;
    double soma_acuracia_classificacao = 0.0;
    vector<vector<int>> matriz_confusao_global(4, vector<int>(4, 0));

    for (int k = 0; k < K_FOLDS; k++) {
        vector<vector<double>> X_fold_treino, X_fold_val;
        vector<double> Y_fold_treino, Y_fold_val; // Para Regressão
        vector<double> Z_fold_treino, Z_fold_val; // Para Classificação

        int testStart = k * foldSize;
        int testEnd = (k == K_FOLDS - 1) ? X.size() : (k + 1) * foldSize;

        for (int i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            if (i >= testStart && i < testEnd) {
                X_fold_val.push_back(X[idx]);
                Y_fold_val.push_back(Y[idx]);
                Z_fold_val.push_back(Z_double[idx]);
            } else {
                X_fold_treino.push_back(X[idx]);
                Y_fold_treino.push_back(Y[idx]);
                Z_fold_treino.push_back(Z_double[idx]);
            }
        }

        // =======================================================
        // TREINAMENTO DOS DOIS MODELOS SEPARADOS
        // =======================================================
        Forest regForest(NUM_TREES, REGRESSION, X_fold_treino, Y_fold_treino, generator, TREE_DEPTH);
        Forest classForest(NUM_TREES, CLASSIFICATION, X_fold_treino, Z_fold_treino, generator, TREE_DEPTH);

        // =======================================================
        // AVALIAÇÃO NESTE FOLD
        // =======================================================
        int acertos_fold = 0;
        double soma_erro_quad_fold = 0.0;
        
        for (int i = 0; i < X_fold_val.size(); i++) {
            
            // --- 1. AVALIANDO A FLORESTA DE REGRESSÃO ---
            // A classe já sabe que tem que fazer a média e devolver a gravidade!
            double grav_prevista = regForest.predict(X_fold_val[i]);
            soma_erro_quad_fold += pow(grav_prevista - Y_fold_val[i], 2);

            // --- 2. AVALIANDO A FLORESTA DE CLASSIFICAÇÃO ---
            // A classe já sabe que tem que fazer a votação e devolver o vencedor!
            int class_prevista = classForest.predict(X_fold_val[i]);
            int class_real = round(Z_fold_val[i]);

            if (class_prevista == class_real) acertos_fold++;
            
            matriz_confusao_global[class_real - 1][class_prevista - 1]++;
        }

        double rmse_fold = sqrt(soma_erro_quad_fold / X_fold_val.size());
        double acc_fold = ((double)acertos_fold / X_fold_val.size()) * 100.0;
        
        cout << "Fold " << (k+1) << " -> RMSE Regressao: " << rmse_fold << " | Acc Classificacao: " << acc_fold << "%" << endl;
        
        soma_rmse_regressao += rmse_fold;
        soma_acuracia_classificacao += acc_fold;
    }

    // =========================================================================
    // EXPORTAÇÃO
    // =========================================================================
    double media_rmse = soma_rmse_regressao / K_FOLDS;
    double media_acc = soma_acuracia_classificacao / K_FOLDS;

    cout << "\nExportando metricas para o script Python..." << endl;

    ofstream outMatrix("confusion_matrix_forest.csv");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) outMatrix << matriz_confusao_global[i][j] << (j == 3 ? "" : ",");
        outMatrix << "\n";
    }
    outMatrix.close();

    ofstream outMetrics("metrics_forest.csv");
    outMetrics << "metric,class,value\n";
    outMetrics << "RMSE,all," << media_rmse << "\n";
    outMetrics << "Accuracy,all," << media_acc << "\n";

    for (int i = 0; i < 4; i++) {
        int TP = matriz_confusao_global[i][i];
        int FP = 0; for (int j = 0; j < 4; j++) FP += matriz_confusao_global[j][i]; FP -= TP;
        int FN = 0; for (int j = 0; j < 4; j++) FN += matriz_confusao_global[i][j]; FN -= TP;

        double precision = (TP + FP == 0) ? 0 : (double)TP / (TP + FP);
        double recall = (TP + FN == 0) ? 0 : (double)TP / (TP + FN);
        double f1 = (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);

        outMetrics << "Precision," << (i + 1) << "," << precision << "\n";
        outMetrics << "Recall," << (i + 1) << "," << recall << "\n";
        outMetrics << "F1-Measure," << (i + 1) << "," << f1 << "\n";
    }
    outMetrics.close();

    return 0;
}