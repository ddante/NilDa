project(unitTests_saveLoad)

# Create the executable
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
    nilda::primitives
    nilda::utils
    nilda::nn
    nilda::layers
    nilda::loss
    nilda::opt
    Eigen3::Eigen
)
