# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/599/bin/cmake

# The command to remove a file.
RM = /snap/cmake/599/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dante/dev/NilDa

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dante/dev/NilDa/Release

# Include any dependencies generated for this target.
include sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/depend.make

# Include the progress variables for this target.
include sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/progress.make

# Include the compile flags for this target's objects.
include sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/flags.make

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/sgd.cpp.o: sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/flags.make
sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/sgd.cpp.o: ../sources/core/neuralNetwork/optimizers/sgd.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/sgd.cpp.o"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opt.dir/sgd.cpp.o -c /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/sgd.cpp

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/sgd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opt.dir/sgd.cpp.i"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/sgd.cpp > CMakeFiles/opt.dir/sgd.cpp.i

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/sgd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opt.dir/sgd.cpp.s"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/sgd.cpp -o CMakeFiles/opt.dir/sgd.cpp.s

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adaGrad.cpp.o: sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/flags.make
sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adaGrad.cpp.o: ../sources/core/neuralNetwork/optimizers/adaGrad.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adaGrad.cpp.o"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opt.dir/adaGrad.cpp.o -c /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/adaGrad.cpp

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adaGrad.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opt.dir/adaGrad.cpp.i"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/adaGrad.cpp > CMakeFiles/opt.dir/adaGrad.cpp.i

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adaGrad.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opt.dir/adaGrad.cpp.s"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/adaGrad.cpp -o CMakeFiles/opt.dir/adaGrad.cpp.s

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/rmsProp.cpp.o: sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/flags.make
sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/rmsProp.cpp.o: ../sources/core/neuralNetwork/optimizers/rmsProp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/rmsProp.cpp.o"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opt.dir/rmsProp.cpp.o -c /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/rmsProp.cpp

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/rmsProp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opt.dir/rmsProp.cpp.i"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/rmsProp.cpp > CMakeFiles/opt.dir/rmsProp.cpp.i

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/rmsProp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opt.dir/rmsProp.cpp.s"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/rmsProp.cpp -o CMakeFiles/opt.dir/rmsProp.cpp.s

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adam.cpp.o: sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/flags.make
sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adam.cpp.o: ../sources/core/neuralNetwork/optimizers/adam.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adam.cpp.o"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opt.dir/adam.cpp.o -c /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/adam.cpp

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adam.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opt.dir/adam.cpp.i"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/adam.cpp > CMakeFiles/opt.dir/adam.cpp.i

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adam.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opt.dir/adam.cpp.s"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers/adam.cpp -o CMakeFiles/opt.dir/adam.cpp.s

# Object files for target opt
opt_OBJECTS = \
"CMakeFiles/opt.dir/sgd.cpp.o" \
"CMakeFiles/opt.dir/adaGrad.cpp.o" \
"CMakeFiles/opt.dir/rmsProp.cpp.o" \
"CMakeFiles/opt.dir/adam.cpp.o"

# External object files for target opt
opt_EXTERNAL_OBJECTS =

sources/core/neuralNetwork/optimizers/libopt.a: sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/sgd.cpp.o
sources/core/neuralNetwork/optimizers/libopt.a: sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adaGrad.cpp.o
sources/core/neuralNetwork/optimizers/libopt.a: sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/rmsProp.cpp.o
sources/core/neuralNetwork/optimizers/libopt.a: sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/adam.cpp.o
sources/core/neuralNetwork/optimizers/libopt.a: sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/build.make
sources/core/neuralNetwork/optimizers/libopt.a: sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library libopt.a"
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && $(CMAKE_COMMAND) -P CMakeFiles/opt.dir/cmake_clean_target.cmake
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/build: sources/core/neuralNetwork/optimizers/libopt.a

.PHONY : sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/build

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/clean:
	cd /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers && $(CMAKE_COMMAND) -P CMakeFiles/opt.dir/cmake_clean.cmake
.PHONY : sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/clean

sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/depend:
	cd /home/dante/dev/NilDa/Release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dante/dev/NilDa /home/dante/dev/NilDa/sources/core/neuralNetwork/optimizers /home/dante/dev/NilDa/Release /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers /home/dante/dev/NilDa/Release/sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sources/core/neuralNetwork/optimizers/CMakeFiles/opt.dir/depend
