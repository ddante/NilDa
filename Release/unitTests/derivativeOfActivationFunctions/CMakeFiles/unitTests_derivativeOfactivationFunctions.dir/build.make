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
include unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/depend.make

# Include the progress variables for this target.
include unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/progress.make

# Include the compile flags for this target's objects.
include unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/flags.make

unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.o: unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/flags.make
unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.o: ../unitTests/derivativeOfActivationFunctions/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.o"
	cd /home/dante/dev/NilDa/Release/unitTests/derivativeOfActivationFunctions && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.o -c /home/dante/dev/NilDa/unitTests/derivativeOfActivationFunctions/main.cpp

unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.i"
	cd /home/dante/dev/NilDa/Release/unitTests/derivativeOfActivationFunctions && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dante/dev/NilDa/unitTests/derivativeOfActivationFunctions/main.cpp > CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.i

unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.s"
	cd /home/dante/dev/NilDa/Release/unitTests/derivativeOfActivationFunctions && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dante/dev/NilDa/unitTests/derivativeOfActivationFunctions/main.cpp -o CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.s

# Object files for target unitTests_derivativeOfactivationFunctions
unitTests_derivativeOfactivationFunctions_OBJECTS = \
"CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.o"

# External object files for target unitTests_derivativeOfactivationFunctions
unitTests_derivativeOfactivationFunctions_EXTERNAL_OBJECTS =

unitTests/derivativeOfActivationFunctions/unitTests_derivativeOfactivationFunctions: unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/main.cpp.o
unitTests/derivativeOfActivationFunctions/unitTests_derivativeOfactivationFunctions: unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/build.make
unitTests/derivativeOfActivationFunctions/unitTests_derivativeOfactivationFunctions: sources/core/neuralNetwork/lossFunctions/libloss.a
unitTests/derivativeOfActivationFunctions/unitTests_derivativeOfactivationFunctions: sources/core/neuralNetwork/libnn.a
unitTests/derivativeOfActivationFunctions/unitTests_derivativeOfactivationFunctions: sources/core/neuralNetwork/layers/liblayers.a
unitTests/derivativeOfActivationFunctions/unitTests_derivativeOfactivationFunctions: unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable unitTests_derivativeOfactivationFunctions"
	cd /home/dante/dev/NilDa/Release/unitTests/derivativeOfActivationFunctions && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/build: unitTests/derivativeOfActivationFunctions/unitTests_derivativeOfactivationFunctions

.PHONY : unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/build

unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/clean:
	cd /home/dante/dev/NilDa/Release/unitTests/derivativeOfActivationFunctions && $(CMAKE_COMMAND) -P CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/cmake_clean.cmake
.PHONY : unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/clean

unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/depend:
	cd /home/dante/dev/NilDa/Release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dante/dev/NilDa /home/dante/dev/NilDa/unitTests/derivativeOfActivationFunctions /home/dante/dev/NilDa/Release /home/dante/dev/NilDa/Release/unitTests/derivativeOfActivationFunctions /home/dante/dev/NilDa/Release/unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : unitTests/derivativeOfActivationFunctions/CMakeFiles/unitTests_derivativeOfactivationFunctions.dir/depend

