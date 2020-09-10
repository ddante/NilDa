project(example_nndense_MNIST)

# Create the executable
add_executable(${PROJECT_NAME} mnist.cpp)

target_link_libraries(${PROJECT_NAME}
    nilda::primitives
    nilda::utils
    nilda::nn
    nilda::layers
    nilda::loss
    nilda::opt
    Eigen3::Eigen    
)
