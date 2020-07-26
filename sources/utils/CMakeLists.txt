add_library(utils
  importMNISTDatasets.cpp
)

add_library(nilda::utils ALIAS utils)

target_include_directories(utils
                           PUBLIC ${PROJECT_SOURCE_DIR}/sources
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                          )
