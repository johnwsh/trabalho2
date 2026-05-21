#include "network.h"
#include <cmath>
#include <iostream>

Network::Network(vector<int> topology, mt19937& generator, NetType type) {
    int numLayers = topology.size();
    this->type = type;
    this->layers.push_back(Layer(topology[0], 0, INPUT, generator));

    for (int i = 1; i < numLayers - 1; i++) {
        int numNeurons = topology[i];
        int numInputs = topology[i - 1]; 
        
        this->layers.push_back(Layer(numNeurons, numInputs, HIDDEN, generator));
    }

    int outputIndex = numLayers - 1;
    int numOutputNeurons = topology[outputIndex];
    int numOutputInputs = topology[outputIndex - 1];

    this->layers.push_back(Layer(numOutputNeurons, numOutputInputs, OUTPUT, generator));
}

void Network::predict(const vector<double>& sample) {
    this->layers[0].forward(sample, this->type);

    for (int i = 1; i < this->layers.size(); i++) {
        this->layers[i].forward(this->layers[i - 1].output, this->type);
    }

    this->output = this->layers.back().output;
}

void Network::computeErrors(const vector<double>& expectedOutput) {

    Layer& outputLayer = this->layers.back(); 
    
    for (int i = 0; i < outputLayer.neuronsLayer.size(); i++) {
        //erro bruto do output
        Neuron& n = outputLayer.neuronsLayer[i];
        
        double error = expectedOutput[i] - n.output;
        
        if (this->type == REGRESSION) {
            n.delta = error; 
        } else {
            //derivada sigmoid
            n.delta = error * (n.output * (1.0 - n.output));
        }
    }

    for (int l = this->layers.size() - 2; l > 0; l--) {
        //erro das camadas ocultas
        Layer& currentLayer = this->layers[l];
        Layer& nextLayer = this->layers[l + 1]; 
        
        for (int i = 0; i < currentLayer.neuronsLayer.size(); i++) {
            Neuron& n = currentLayer.neuronsLayer[i];
            double somaDosErros = 0.0;
            
            for (int j = 0; j < nextLayer.neuronsLayer.size(); j++) {
                Neuron& nextNode = nextLayer.neuronsLayer[j];
                
                somaDosErros += nextNode.weights[i] * nextNode.delta;
            }
            //derivada sigmoid
            n.delta = somaDosErros * (n.output * (1.0 - n.output));
        }
    }
}

void Network::updateWeights(double learningRate, double momentumFactor) {
    
    for (int l = 1; l < this->layers.size(); l++) {
        
        vector<double>& inputsDaCamada = this->layers[l - 1].output;

        for (int i = 0; i < this->layers[l].neuronsLayer.size(); i++) {
            
            this->layers[l].neuronsLayer[i].updateWeights(inputsDaCamada, learningRate, momentumFactor);
        }
    }
}

void Network::train(const vector<vector<double>>& X, const vector<vector<double>>& Y, double learningRate, double momentumFactor, int maxEpochs, double minError) {
    
    double erroAnterior = 999999.0; 
    double erroAtual = 0.0;
    int epoca = 0;

    while (epoca < maxEpochs) {
        erroAtual = 0.0;

        for (int i = 0; i < X.size(); i++) {
            
            //feedforward
            this->predict(X[i]);
            
            //Erro MSE
            for(int j = 0; j < Y[i].size(); j++) {
                erroAtual += pow(Y[i][j] - this->output[j], 2);
            }

            this->computeErrors(Y[i]);
            
            this->updateWeights(learningRate, momentumFactor);
        }

        erroAtual = erroAtual / X.size();

        if (epoca % 100 == 0) {
            cout << "Epoca: " << epoca << " | Erro Atual: " << erroAtual << endl;
        }

        if (abs(erroAnterior - erroAtual) <= minError) {
            break; 
        }

        erroAnterior = erroAtual;
        epoca++;
    }
}

vector<vector<double>> oneHotEncode(const vector<double>& labels, int numClasses) {
    //Para transformar uma saida de uma classe em um vetor tipo {0,0,1,0}
    vector<vector<double>> encodedMatrix;

    for (int i = 0; i < labels.size(); i++) {
        vector<double> oneHot(numClasses, 0.0);

        int classe_paciente = static_cast<int>(labels[i]);
        int indice = classe_paciente - 1;

        oneHot[indice] = 1.0;

        encodedMatrix.push_back(oneHot);
    }

    return encodedMatrix;
}

vector<vector<double>> normalize(const vector<vector<double>>& X, vector<double>& minVals, vector<double>& maxVals) {
    
    if (X.empty() || X[0].empty()) return X; 

    int numPacientes = X.size();
    int numFeatures = X[0].size();

    minVals.assign(numFeatures, numeric_limits<double>::max());
    maxVals.assign(numFeatures, numeric_limits<double>::lowest());

    for (int i = 0; i < numPacientes; i++) {
        for (int j = 0; j < numFeatures; j++) {
            if (X[i][j] < minVals[j]) minVals[j] = X[i][j];
            if (X[i][j] > maxVals[j]) maxVals[j] = X[i][j];
        }
    }

    vector<vector<double>> X_normalizado = X; 

    for (int i = 0; i < numPacientes; i++) {
        for (int j = 0; j < numFeatures; j++) {
            
            if (maxVals[j] == minVals[j]) {
                X_normalizado[i][j] = 0.0; 
            } else {
                X_normalizado[i][j] = (X[i][j] - minVals[j]) / (maxVals[j] - minVals[j]);
            }
        }
    }

    return X_normalizado;
}