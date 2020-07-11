add_library(nn                
                layer.cpp
                inputLayer.cpp
                denseLayer.cpp
                neuralNetwork.cpp)

add_library(nilda::nn ALIAS nn)

target_include_directories( nn
                                     PUBLIC ${PROJECT_SOURCE_DIR}/sources
                                     INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                                            
)
