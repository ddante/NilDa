subdirs(
    layers
    lossFunctions
    optimizers
)

add_library(nn
            neuralNetwork.cpp)

add_library(nilda::nn ALIAS nn)

#find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
#link_directories(${HDF5_LIBRARY_DIRS})
#include_directories(${HDF5_INCLUDE_DIR})

target_include_directories(nn
                           PUBLIC ${PROJECT_SOURCE_DIR}/sources
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                          )

#target_link_libraries(nn ${HDF5_CXX_LIBRARIES})
