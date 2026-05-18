#ifndef FOREST_H
#define FOREST_H
#include "trees.h"
#include <vector>

using namespace std;

class Forest {
    public:
        TreeType type;
        vector<Tree*> forest;

        Forest(int numTrees, TreeType type, vector<vector<double>> &X, vector<double> &Y, mt19937 &generator, int treeDepth);
        ~Forest(){
            for (int i = 0; i< forest.size(); i++){
                delete forest[i];
            }
        }
        double predict(vector<double>& sample);
};


#endif