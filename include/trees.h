#ifndef TREES_H
#define TREES_H

#include <vector>

using namespace std;

enum TreeType {
    CLASSIFICATION,
    REGRESSION
};

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
        double predict(const vector<double>& sample);
};

Node* CART(int maxDepth, 
           int minSamplesSplit, 
           TreeType type, 
           const vector<vector<double>>& X, 
           const vector<double>& y, 
           int currentDepth = 0);

//Utilizado para abstração
Tree* buildTree(int maxDepth, int minSamplesSplit, TreeType type, 
                const vector<vector<double>>& X, 
                const vector<double>& y);

#endif