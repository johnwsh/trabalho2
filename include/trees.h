#ifndef TREES_H
#define TREES_H

#include <vector>
#include <random>

using namespace std;

enum TreeType {
    CLASSIFICATION,
    REGRESSION
};

void buildBootstrappedDataset(vector<vector<double>>& X, 
                                                vector<double>& Y, 
                                                int numSamples, 
                                                mt19937 &generator, 
                                                vector<vector<double>>& newDatasetX, 
                                                vector<double>& newDatasetY);

struct Node {
    int featureIndex = -1; //índice do dataset que indica sobre qual feature é esse nó
    double threshold = 0.0; //valor de corte para decidir para onde vai, se sample[featureindex] <= threshold, então é verdadeiro (vai pra esquerda), senão é falso (vai pra direita)
    Node* left = nullptr;
    Node* right = nullptr;
    double value = 0.0; //se o nó for uma folha, aqui vai ter o valor predito
    bool isLeaf = false;

    ~Node() {
        delete left;
        delete right;
    }
};

class Tree {
    public:
        TreeType type;
        Node* root;

        Tree(TreeType t) : type(t), root(nullptr) {}

        ~Tree() { 
            delete root; 
        }

        //dado um vetor com features, retorna a predição (Classe ou valor)
        double predict(vector<double>& sample);
};

Node* CART(int maxDepth, 
           int minSamplesSplit, 
           vector<vector<double>>& X, 
           vector<double>& y, 
           mt19937 &generator,
           int currentDepth = 0);

//Utilizado para abstração
Tree* buildTree(int maxDepth, int minSamplesSplit, TreeType type, 
                vector<vector<double>>& X, 
                vector<double>& y);

double calculateMSE(int attIndex, vector<vector<double>> &X, vector<double> &Y, double &threshold, mt19937 &generator);

#endif