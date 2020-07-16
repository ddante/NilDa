add_library(loss
                sparseCategoricalCrossentropy.cpp
               )

add_library(nilda::loss ALIAS loss)

target_include_directories(loss
                                    PUBLIC ${PROJECT_SOURCE_DIR}/sources
                                    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}                                           
                                   )

