add_library(layers
            inputLayer.cpp
            denseLayer.cpp
            conv2DLayer.cpp
            conv2DUtils.cpp
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
