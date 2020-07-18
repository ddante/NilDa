add_library(primitives INTERFACE)

add_library(nilda::primitives ALIAS primitives)

target_include_directories(primitives
                                  INTERFACE 
                                  ${CMAKE_CURRENT_SOURCE_DIR}
                                  )
