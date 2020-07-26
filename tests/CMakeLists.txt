project(tests)

# Create the executable
add_executable(${PROJECT_NAME} main.cpp)

# Link the static library from subproject1 using it's alias sub::lib1
# Link the header only library from subproject2 using it's alias sub::lib2
# This will cause the include directories for that target to be added to this project
target_link_libraries(${PROJECT_NAME}
    nilda::primitives
    nilda::utils
    nilda::nn
    nilda::layers
    nilda::loss
    nilda::opt
    Eigen3::Eigen
)
