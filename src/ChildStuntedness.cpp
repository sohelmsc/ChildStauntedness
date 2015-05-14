/*
 *  ChildStuntedness.cpp

 *
 *  Created on: Aug 16, 2014
 *      Author: Mafijul Bhuiyan
 *
 */

#include <cmath>
#include <vector>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

using std::vector;
using namespace std;

typedef std::pair<int,int> idPair;

struct comparator
{
	bool operator () (const idPair& left, const idPair& right)
	{
		return left.first < right.first;
	}
};


class ChildStuntedness{

	public:
			vector<double> predict(vector<string> &sTraining, vector<string> &sTesting);
	private:
			 inline void weightUpdate(double *weights, double *nodeVal,double **inputData, double out, double learnRate, int inputNum, int nodeNum,int hidLayers, int rowNum);
			 inline void computeNeuron(double **trainData, double *weights, double *nodeVal, int hidLayers, int inputNum, int rowNum, int param);
			 inline double activationFunction(double x, int param);
};


// *** Predict the output parameters based on the training of the artificial neural network (ANN)
// *** Here back propagation multi-layer perceptron (MLP) is used for training the input dataset.
vector<double> ChildStuntedness::predict(vector<string> &sTraining, vector<string> &sTesting)
{
	double *weights, *weights1, learningRate[2]={0.1, 0.1}, totErr, error, *nodeVal, *nodeVal1;
	double **inputTesting, **inputTraining, **outputTraining, maxError=10.0E-15;
    int hiddenLayers=2, dataNum, dataCountTrain, elementCount, dataCountTest, epochNum=700,epochCount=0;
    int weightNum, inputNum=8, nodeNum, babyId, dataNum1;
    std::string token;
    vector<double>b;
    vector<idPair> id;

    dataNum = sTraining.size();
    dataNum1 = sTesting.size();

    /* Considering 8 input parameters */
    weightNum = (inputNum+1) * (hiddenLayers*inputNum + 1);
    nodeNum = (hiddenLayers*inputNum +  hiddenLayers +1);

    /*  1D double dynamic memory allocation with ( weightNum X 1) size ************/
	weights = (double *) malloc(sizeof(double) * weightNum);
	weights1 = (double *) malloc(sizeof(double) * weightNum);

	/*  1D double dynamic memory allocation with ( Num X 1) size *****************/
	nodeVal = (double *) malloc(sizeof(double) * nodeNum);
	nodeVal1 = (double *) malloc(sizeof(double) * nodeNum);

    /*  2D double dynamic memory allocation with (dataNum X tokenNum+1) size *****/
    inputTraining = (double **) malloc(sizeof(double *) * dataNum);
    for(int i=0; i<dataNum; i++)
    	inputTraining[i] = (double *) malloc(sizeof(double) * inputNum+1);

    /*  2D double dynamic memory allocation with (dataNum X 2) size *************/
	outputTraining = (double **) malloc(sizeof(double *) * dataNum);
	for(int i=0; i<dataNum; i++)
		outputTraining[i] = (double *) malloc(sizeof(double) * 2);

	/*  2D double dynamic memory allocation with (dataNum X tokenNum+1) size ************/
	inputTesting = (double **) malloc(sizeof(double *) * dataNum1);
	for(int i=0; i<dataNum1; i++)
		inputTesting[i] = (double *) malloc(sizeof(double) * inputNum+1);


	/****************************** Initialise random weights  ***********************/
	for(int i=0; i<weightNum; i++)
		weights[i] = rand()/(double)(RAND_MAX+1.0);

	for(int i=0; i<weightNum; i++)
		weights1[i] = rand()/(double)(RAND_MAX+1.0);

	/****************************** Parsing the training data set *******************/
	dataCountTrain = 0;
    while(dataCountTrain<dataNum)
	{
		std::istringstream fetusInfo(sTraining[dataCountTrain]);
		elementCount = 0;
    	while(getline(fetusInfo, token, ','))
		{
    		if(elementCount > 3)
    		{
				std::istringstream temp(token);
				if(elementCount>inputNum+4-1)
					temp >> outputTraining[dataCountTrain][elementCount - (inputNum+4)];
				else
					temp >> inputTraining[dataCountTrain][elementCount-4];
    		}
			elementCount++;
		}
    	/* BIAS of the network */
    	inputTraining[dataCountTrain][inputNum] = -1;
    	dataCountTrain++;
	 }

	/****************************** Parsing the testing data set *********************/
	dataCountTest = 0;
    while(dataCountTest<dataNum1)
	{
		std::istringstream fetusInfo(sTesting[dataCountTest]);
		elementCount = 0;
    	while(getline(fetusInfo, token, ','))
		{
    		if(elementCount>3)
    		{
				std::istringstream temp(token);
				temp >> inputTesting[dataCountTest][elementCount-4];
    		}
    		else if(elementCount == 0)
    		{
    			std::istringstream temp(token);
    			temp >>babyId;
    			id.push_back(idPair(babyId, dataCountTest));
    		}
			elementCount++;
		}

    	/* BIAS of the network */
    	inputTesting[dataCountTest][inputNum] = -1;
    	dataCountTest++;
	}

	/***Train the Multi-layer perceptron (MLP) network using back-propagation (BP) Algo *****/
    totErr = 10000.0;
    while((epochCount < epochNum) && (totErr-maxError)>0.0)
    {
    	totErr = 0;
    	for(int i=0; i<dataNum; i++)
    	{
    		computeNeuron(inputTraining, weights, nodeVal, hiddenLayers,inputNum, i, 0);
    		error = 0.5*pow((outputTraining[i][0] - nodeVal[nodeNum-1]),2);
    		weightUpdate(weights,nodeVal,inputTraining, outputTraining[i][0],learningRate[0],inputNum, nodeNum, hiddenLayers,i);
    		totErr+=error;
    	}
    	epochCount++;
    }
    totErr = 10000.0;
    epochCount = 0;
    while((epochCount < epochNum) && (totErr-maxError)>0.0)
    {
    	totErr = 0;
    	for(int i=0; i<dataNum; i++)
    	{
    		computeNeuron(inputTraining, weights1, nodeVal1, hiddenLayers,inputNum, i, 1);
    		error = 0.5*pow((outputTraining[i][1] - nodeVal1[nodeNum-1]),2);
    		weightUpdate(weights1,nodeVal1,inputTraining, outputTraining[i][1],learningRate[1],inputNum, nodeNum, hiddenLayers,i);
    		totErr+=error;
    	}
    	epochCount++;
    }

    std::sort(id.begin(), id.end(), comparator());

    int pre_id = id[0].first;

    for(int i=0; i<=dataNum1; i++)
	{

		if((i>0 && pre_id != id[i].first))
		{
			computeNeuron(inputTesting, weights, nodeVal, hiddenLayers,inputNum, id[i].second, 0);
			b.push_back(nodeVal[nodeNum-1]);

			// Baby birth duration
			computeNeuron(inputTesting, weights1, nodeVal1, hiddenLayers,inputNum, id[i].second, 1);
			b.push_back(nodeVal1[nodeNum-1]);

		}

		else if(i == 0)
		{
			// Baby weight
			computeNeuron(inputTesting, weights, nodeVal, hiddenLayers,inputNum, id[i].second, 0);
			b.push_back(nodeVal[nodeNum-1]);

			// Baby birth duration
			computeNeuron(inputTesting, weights1, nodeVal1, hiddenLayers,inputNum, id[i].second, 1);
			b.push_back(nodeVal1[nodeNum-1]);
		}
		pre_id = id[i].first;
	}

	return b;
}


//** Updating the weight of each neuron in the neural network **************************************//
inline void ChildStuntedness::weightUpdate(double *weights, double *nodeVal, double **inputData, double out, double learnRate, int inputNum, int nodeNum, int hidLayers, int rowNum)
{
	double gamma[inputNum], preGamma[inputNum], temp;
	int gammaCount=0, preGammaCount;

	gamma[gammaCount] = nodeVal[nodeNum-1] *(1 - nodeVal[nodeNum-1]) * (out - nodeVal[nodeNum-1]);
	gammaCount++;
	preGammaCount = gammaCount;
	for(int i=0; i<inputNum; i++)
		preGamma[i] = gamma[i];

	/* Last layer weight update */
	for(int j=0; j<=inputNum; j++)
		weights[(hidLayers)*inputNum*(inputNum+1)+j] += learnRate * gamma[gammaCount] * nodeVal[(hidLayers-1)*(inputNum+1)+j];

	/* Weights update using back propagation */
	for(int i=hidLayers-1; i>=0; i--)
	{
		gammaCount = 0;
		for(int k=0; k<inputNum; k++)
		{
			temp = nodeVal[i*(inputNum+1)+k] * (1 - nodeVal[i*(inputNum+1)+k]);
			for(int l=0; l<preGammaCount; l++)
			{
				gamma[k] += weights[(i+1)*inputNum*(inputNum+1)+k+l*(inputNum+1)] * preGamma[l];
			}
			gamma[k] *= temp;
			gammaCount++;
			for(int j=0; j<=inputNum; j++)
			{
				if((i-1) < 0)
					weights[i*inputNum*(inputNum+1)+k*(inputNum+1)+j] += learnRate * gamma[k] * inputData[rowNum][j];
				else
					weights[i*inputNum*(inputNum+1)+k*(inputNum+1)+j] += learnRate * gamma[k] * nodeVal[(i-1)*(inputNum+1)+j];
			}
		}
		preGammaCount = gammaCount;
		for(int i=0; i<inputNum; i++)
			preGamma[i] = gamma[i];
	}
}

// Compute the final output of all neurons **************************************************************//
inline void ChildStuntedness::computeNeuron(double **trainData, double *weights, double *nodeVal, int hidLayers, int inputNum, int rowNum, int param)
{
	double sum;
	int count=0;

	/*Flow:  Inputs ----> Neuron -----> Layers */
	/**Number of hidden layers in the network **/
	for(int i=0; i<hidLayers; i++)
	{
		/**Number of nodes/neurons in a single layer **/
		for(int k=0; k<inputNum; k++)
		{
			sum = 0.0;
			/**Number of inputs in a single neuron **/
			for(int j=0; j<=inputNum; j++)
			{
				if(i>0)
					sum += nodeVal[j+i*(inputNum+1)] * weights[j+k*(inputNum+1)+i*inputNum*(inputNum+1)];
				else
					sum += trainData[rowNum][j] * weights[j+k*(inputNum+1)+i*inputNum*(inputNum+1)];
			}
			nodeVal[count] = activationFunction(sum, param);
			count++;
		}
		nodeVal[count] = -1;
		count++;
	}

	/*  Output neuron computation */
	sum = 0.0;
	for(int j=0; j<=inputNum; j++)
		sum += nodeVal[j+(hidLayers-1)*(inputNum+1)] * weights[j+(hidLayers)*inputNum*(inputNum+1)];

	nodeVal[count] = activationFunction(sum, param);
}

/**************** Used Sigmoid as activation function **********/
inline double ChildStuntedness::activationFunction(double x, int param)
{
	double lambda[2]={0.3,0.009}, p;
	if(param == 0)
		p = 1/(1+exp(-lambda[0]*x));
	else
		p = 1/(1+exp(-lambda[1]*x));
	return p;
}

int main()
{

	vector<string> sTr;
	vector<string> sTs;
	vector<double> b;

	clock_t start, finish;

	/// Sample input parameters
	sTr.push_back("9,0.645051195,0,2,0,0.565724789,0.509087642,0.610132359,0.479853315,0.506471927,0.558087248,0.028567054,0.722222222,0.442367601");
	sTr.push_back("51,0.436860068,1,2,0,0.393117519,0.328628866,0.410235079,0.318274877,0.328387182,0.366602529,0.019592691,0.55952381,0.414330218");
	sTr.push_back("64,0.529010239,1,2,0,0.497251757,0.406787932,0.529352757,0.406420842,0.398363189,0.463302586,0.02557803,0.660714286,0.429906542");

	sTs.push_back("12,0.341296928,0,2,0,0.269644443,0.219533454,0.30020437,0.21132084,0.222165281,0.182502559,0.014711833");
	sTs.push_back("12,0.457337884,0,2,0,0.426052467,0.381421198,0.456616876,0.369614615,0.389651129,0.376981331,0.021929233");
	sTs.push_back("39,0.412969283,1,2,0,0.343526018,0.275821967,0.364040532,0.263627978,0.283522405,0.274536333,0.017267084");
	sTs.push_back("14,0.627986348,0,2,0,0.606170136,0.567277203,0.654352757,0.567556825,0.561811363,0.625492722,0.031512337");

    ChildStuntedness ch;
    start = clock();
	b = ch.predict(sTr,sTs);
	finish = clock();
    std::cout << "Time: " << (finish-start)/double(CLOCKS_PER_SEC) << " Seconds " <<std::endl;

	for(int i=0; i<b.size(); i++)
			std::cout << b[i] << "-----" <<std::endl;
    return 0;
}
