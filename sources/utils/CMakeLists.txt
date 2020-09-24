add_library(utils
  Random.cpp
  importMNISTDatasets.cpp
  images.cpp
)

add_library(nilda::utils ALIAS utils)

target_include_directories(utils
                           PUBLIC ${PROJECT_SOURCE_DIR}/sources
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                          )

target_link_libraries(utils ${OpenCV_LIBS})
