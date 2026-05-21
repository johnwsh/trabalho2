#include "../include/forest.h"
#include <cmath>
#include <map>

Forest::Forest(int numTrees, TreeType type, vector<vector<double>> &X, vector<double> &Y, mt19937 &generator, int treeDepth){
    this->type = type;
    for (int i = 0; i < numTrees; i++){
        
        vector<vector<double>> Xb;
        vector<double> Yb;

        vector<bool> usedSamples(X.size(), false);

        buildBootstrappedDataset(X,Y,X.size(),generator,Xb,Yb, usedSamples);
        Tree* tree = buildTree(treeDepth, type, Xb, Yb, usedSamples, generator);
        forest.push_back(tree);
    }
}


double Forest::predict(vector<double>& sample, int sampleIndex) {
    
    if (this->type == REGRESSION) {
        double sumPredict = 0.0;
        int usedTrees = 0;
        
        for (int i = 0; i < forest.size(); i++) {
            if (sampleIndex == -1 || forest[i]->usedSamples[sampleIndex]) {
                sumPredict += forest[i]->predict(sample);
                usedTrees++;
            }
        }
        
        return (usedTrees > 0) ? (sumPredict / usedTrees) : 0.0; 
    }
    else { 
        map<int, int> votos;
        int classe_vencedora = -1;
        int max_votos = -1;

        for (int i = 0; i < forest.size(); i++) {
            
            if (sampleIndex == -1 || forest[i]->usedSamples[sampleIndex]) {
                
                int voto_da_arvore = round(forest[i]->predict(sample));
                votos[voto_da_arvore]++;

                if (votos[voto_da_arvore] > max_votos) {
                    max_votos = votos[voto_da_arvore];
                    classe_vencedora = voto_da_arvore;
                }
            }
        }
        
        return classe_vencedora;
    }
}