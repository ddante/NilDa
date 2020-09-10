#include <iostream>
#include <math.h>
#include "primitives/errors.h"
#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"

const NilDa::Scalar errTol = 1e-10;

int test1()
{
    NilDa::layer* l0 = new NilDa::inputLayer(3);
    NilDa::layer* l1 = new NilDa::denseLayer(2, "relu");
    NilDa::layer* l2 = new NilDa::denseLayer(3, "softmax");

    NilDa::neuralNetwork nn({l0, l1, l2});

    NilDa::Matrix trainingData(3, 4);
    trainingData <<  1.1, 1.21, 0.2, 0.42,
                     0.3, 0.63, 4.4, 4.84,
                     5.1, -0.3, 0.9, -1.4;

    NilDa::Matrix trainingLabels(3, 4);
    trainingLabels << 1,0,0,0,
                      0,1,0,1,
                      0,0,1,0;

    NilDa::Matrix W1(2,3);
    W1 <<  -1, 2, -3, 0.4, -0.5, -0.6;

    NilDa::Vector b1(2);
    b1 << 0.3, 0.5;

    l1->setWeightsAndBiases(W1, b1);

    NilDa::Matrix W2(3,2);
    W2 << -1, 2, 0.3, 0.4, 0.6, 0.7;

    NilDa::Vector b2(3);
    b2 << 0.3, 0.5, 0.7;

    l2->setWeightsAndBiases(W2, b2);

    nn.setLossFunction("sparse_categorical_crossentropy");

    nn.forwardPropagation(trainingData);
    nn.backwardPropagation(trainingData, trainingLabels);

    NilDa::Matrix dw2 = l2->getWeightsDerivative();
    NilDa::Vector db2 = l2->getBiasesDerivative();

    NilDa::Matrix dw1 = l1->getWeightsDerivative();
    NilDa::Vector db1 = l1->getBiasesDerivative();

    NilDa::Scalar J = nn.getLoss(trainingLabels);

    NilDa::Scalar J_check  = 3.39158288699;

    NilDa::Matrix predictions;

    nn.predict(trainingData, predictions);

    NilDa::Matrix predicitons_check(3,4);
    predicitons_check << 0, 0, 0, 0,
                         0, 0, 0, 0,
                         1, 1, 1, 1;

    NilDa::Matrix dw2_check(3,2);
    dw2_check << 0.0296332550125, 0.0200692970153,
                -6.80027327861, -0.244710363185,
                 6.7706400236,   0.22464106617;

    NilDa::Vector db2_check(3);
    db2_check << -0.237236528562,
                 -0.615960963177,
                  0.853197491739;

    NilDa::Matrix dw1_check(2,3);
    dw1_check << 0.117648199863, 0.672490939748,  -0.237140023208,
                 0.141812563064, 0.0738362931655, -0.0351601396026;

    NilDa::Vector db1_check(2);
    db1_check << 0.179716754353,
                 0.117200465342;

    NilDa::Scalar difference_j = sqrt((J - J_check)*(J - J_check));

    NilDa::Scalar difference_p = (predicitons_check - predictions).norm();

    NilDa::Scalar difference_dw2 = (dw2 - dw2_check).norm();
    NilDa::Scalar difference_db2 = (db2 - db2_check).norm();

    NilDa::Scalar difference_dw1 = (dw1 - dw1_check).norm();
    NilDa::Scalar difference_db1 = (db1 - db1_check).norm();

    std::cout << " Test 1: ";
    if (
        difference_j < errTol &&
        difference_p < errTol &&
        difference_dw1 < errTol &&
        difference_db1 < errTol &&
        difference_dw2 < errTol &&
        difference_db2 < errTol
       )
    {
         std::cout << "test OK\n";
         return NilDa::EXIT_OK;
    }
    else
    {
        std::cout << "test FAILED, differences: ";
        std::cout << difference_j << ", ";
        std::cout << difference_dw1 << ", ";
        std::cout << difference_db1 << ", ";
        std::cout << difference_dw2 << ", ";
        std::cout << difference_db2 << "\n";
        return NilDa::EXIT_FAIL;
    }

}

int main(int argc, char const *argv[])
{
#ifdef ND_SP
    #warning "Single precision used. For testing specify either double or long precision"
#else

    if (test1() == NilDa::EXIT_OK)
    {
        return NilDa::EXIT_OK;
    }
    else
    {
        return NilDa::EXIT_FAIL;
    }

#endif
}
