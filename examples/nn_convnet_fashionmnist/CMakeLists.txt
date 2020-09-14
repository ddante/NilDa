project(example_nnconvnet_fashionMNIST)

# Create the executable
add_executable(${PROJECT_NAME} fashionMNIST.cpp)

target_link_libraries(${PROJECT_NAME}
    nilda::primitives
    nilda::utils
    nilda::nn
    nilda::layers
    nilda::loss
    nilda::opt
    Eigen3::Eigen
)
