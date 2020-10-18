project(unitTests_batchNorm)

# Create the executable
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
    nilda::primitives
    nilda::loss
    nilda::nn
    nilda::layers
    Eigen3::Eigen
)
