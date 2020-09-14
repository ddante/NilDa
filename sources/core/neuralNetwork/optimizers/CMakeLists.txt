add_library(
  opt
  sgd.cpp
  adaGrad.cpp
)
add_library(nilda::opt ALIAS opt)

target_include_directories(opt
                           PUBLIC ${PROJECT_SOURCE_DIR}/sources
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                          )
