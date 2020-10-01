add_library(layers
            layer.cpp
            inputLayer.cpp
            denseLayer.cpp
            dropoutLayer.cpp
            conv2DLayer.cpp
            conv2DUtils.cpp
            maxPool2DLayer.cpp
            pool2DUtils.cpp
            batchNormalizationLayer.cpp
            activationFunctions/activationFunctionUtils.cpp
            activationFunctions/identity.cpp
            activationFunctions/relu.cpp
            activationFunctions/sigmoid.cpp
            activationFunctions/softmax.cpp
           )

add_library(nilda::layers ALIAS layers)

target_include_directories(layers
                           PUBLIC ${PROJECT_SOURCE_DIR}/sources
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                          )
