subdirs(
    layers
    lossFunctions
    optimizers
)

add_library(nn
                neuralNetwork.cpp)

add_library(nilda::nn ALIAS nn)

target_include_directories(nn
                                  PUBLIC ${PROJECT_SOURCE_DIR}/sources
                                  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
