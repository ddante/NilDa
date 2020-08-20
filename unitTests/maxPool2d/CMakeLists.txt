project(unitTests_maxPool2d)

# Create the executable
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
    nilda::primitives
    nilda::nn
    nilda::layers
    nilda::loss
    Eigen3::Eigen
)
