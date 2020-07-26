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
include tests/CMakeFiles/tests.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/tests.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/tests.dir/flags.make

tests/CMakeFiles/tests.dir/main.cpp.o: tests/CMakeFiles/tests.dir/flags.make
tests/CMakeFiles/tests.dir/main.cpp.o: ../tests/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dante/dev/NilDa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/tests.dir/main.cpp.o"
	cd /home/dante/dev/NilDa/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tests.dir/main.cpp.o -c /home/dante/dev/NilDa/tests/main.cpp

tests/CMakeFiles/tests.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tests.dir/main.cpp.i"
	cd /home/dante/dev/NilDa/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dante/dev/NilDa/tests/main.cpp > CMakeFiles/tests.dir/main.cpp.i

tests/CMakeFiles/tests.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tests.dir/main.cpp.s"
	cd /home/dante/dev/NilDa/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dante/dev/NilDa/tests/main.cpp -o CMakeFiles/tests.dir/main.cpp.s

# Object files for target tests
tests_OBJECTS = \
"CMakeFiles/tests.dir/main.cpp.o"

# External object files for target tests
tests_EXTERNAL_OBJECTS =

tests/tests: tests/CMakeFiles/tests.dir/main.cpp.o
tests/tests: tests/CMakeFiles/tests.dir/build.make
tests/tests: sources/utils/libutils.a
tests/tests: sources/core/neuralNetwork/libnn.a
tests/tests: sources/core/neuralNetwork/layers/liblayers.a
tests/tests: sources/core/neuralNetwork/lossFunctions/libloss.a
tests/tests: sources/core/neuralNetwork/optimizers/libopt.a
tests/tests: tests/CMakeFiles/tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dante/dev/NilDa/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tests"
	cd /home/dante/dev/NilDa/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/tests.dir/build: tests/tests

.PHONY : tests/CMakeFiles/tests.dir/build

tests/CMakeFiles/tests.dir/clean:
	cd /home/dante/dev/NilDa/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/tests.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/tests.dir/clean

tests/CMakeFiles/tests.dir/depend:
	cd /home/dante/dev/NilDa/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dante/dev/NilDa /home/dante/dev/NilDa/tests /home/dante/dev/NilDa/build /home/dante/dev/NilDa/build/tests /home/dante/dev/NilDa/build/tests/CMakeFiles/tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/tests.dir/depend

