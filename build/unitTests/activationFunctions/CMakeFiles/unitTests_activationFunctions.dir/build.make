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
CMAKE_COMMAND = /snap/cmake/487/bin/cmake

# The command to remove a file.
RM = /snap/cmake/487/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dante/dev/NilDa

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dante/dev/NilDa/build

# Include any dependencies generated for this target.
include unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/depend.make

# Include the progress variables for this target.
include unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/progress.make

# Include the compile flags for this target's objects.
include unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/flags.make

unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/main.cpp.o: unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/flags.make
unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/main.cpp.o: ../unitTests/activationFunctions/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dante/dev/NilDa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/main.cpp.o"
	cd /home/dante/dev/NilDa/build/unitTests/activationFunctions && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/unitTests_activationFunctions.dir/main.cpp.o -c /home/dante/dev/NilDa/unitTests/activationFunctions/main.cpp

unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/unitTests_activationFunctions.dir/main.cpp.i"
	cd /home/dante/dev/NilDa/build/unitTests/activationFunctions && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dante/dev/NilDa/unitTests/activationFunctions/main.cpp > CMakeFiles/unitTests_activationFunctions.dir/main.cpp.i

unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/unitTests_activationFunctions.dir/main.cpp.s"
	cd /home/dante/dev/NilDa/build/unitTests/activationFunctions && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dante/dev/NilDa/unitTests/activationFunctions/main.cpp -o CMakeFiles/unitTests_activationFunctions.dir/main.cpp.s

# Object files for target unitTests_activationFunctions
unitTests_activationFunctions_OBJECTS = \
"CMakeFiles/unitTests_activationFunctions.dir/main.cpp.o"

# External object files for target unitTests_activationFunctions
unitTests_activationFunctions_EXTERNAL_OBJECTS =

unitTests/activationFunctions/unitTests_activationFunctions: unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/main.cpp.o
unitTests/activationFunctions/unitTests_activationFunctions: unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/build.make
unitTests/activationFunctions/unitTests_activationFunctions: sources/core/neuralNetwork/lossFunctions/libloss.a
unitTests/activationFunctions/unitTests_activationFunctions: sources/core/neuralNetwork/libnn.a
unitTests/activationFunctions/unitTests_activationFunctions: sources/core/neuralNetwork/layers/liblayers.a
unitTests/activationFunctions/unitTests_activationFunctions: unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dante/dev/NilDa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable unitTests_activationFunctions"
	cd /home/dante/dev/NilDa/build/unitTests/activationFunctions && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/unitTests_activationFunctions.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/build: unitTests/activationFunctions/unitTests_activationFunctions

.PHONY : unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/build

unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/clean:
	cd /home/dante/dev/NilDa/build/unitTests/activationFunctions && $(CMAKE_COMMAND) -P CMakeFiles/unitTests_activationFunctions.dir/cmake_clean.cmake
.PHONY : unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/clean

unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/depend:
	cd /home/dante/dev/NilDa/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dante/dev/NilDa /home/dante/dev/NilDa/unitTests/activationFunctions /home/dante/dev/NilDa/build /home/dante/dev/NilDa/build/unitTests/activationFunctions /home/dante/dev/NilDa/build/unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : unitTests/activationFunctions/CMakeFiles/unitTests_activationFunctions.dir/depend

