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
include sources/utils/CMakeFiles/utils.dir/depend.make

# Include the progress variables for this target.
include sources/utils/CMakeFiles/utils.dir/progress.make

# Include the compile flags for this target's objects.
include sources/utils/CMakeFiles/utils.dir/flags.make

sources/utils/CMakeFiles/utils.dir/Random.cpp.o: sources/utils/CMakeFiles/utils.dir/flags.make
sources/utils/CMakeFiles/utils.dir/Random.cpp.o: ../sources/utils/Random.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object sources/utils/CMakeFiles/utils.dir/Random.cpp.o"
	cd /home/dante/dev/NilDa/Release/sources/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/utils.dir/Random.cpp.o -c /home/dante/dev/NilDa/sources/utils/Random.cpp

sources/utils/CMakeFiles/utils.dir/Random.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/utils.dir/Random.cpp.i"
	cd /home/dante/dev/NilDa/Release/sources/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dante/dev/NilDa/sources/utils/Random.cpp > CMakeFiles/utils.dir/Random.cpp.i

sources/utils/CMakeFiles/utils.dir/Random.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/utils.dir/Random.cpp.s"
	cd /home/dante/dev/NilDa/Release/sources/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dante/dev/NilDa/sources/utils/Random.cpp -o CMakeFiles/utils.dir/Random.cpp.s

sources/utils/CMakeFiles/utils.dir/importMNISTDatasets.cpp.o: sources/utils/CMakeFiles/utils.dir/flags.make
sources/utils/CMakeFiles/utils.dir/importMNISTDatasets.cpp.o: ../sources/utils/importMNISTDatasets.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object sources/utils/CMakeFiles/utils.dir/importMNISTDatasets.cpp.o"
	cd /home/dante/dev/NilDa/Release/sources/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/utils.dir/importMNISTDatasets.cpp.o -c /home/dante/dev/NilDa/sources/utils/importMNISTDatasets.cpp

sources/utils/CMakeFiles/utils.dir/importMNISTDatasets.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/utils.dir/importMNISTDatasets.cpp.i"
	cd /home/dante/dev/NilDa/Release/sources/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dante/dev/NilDa/sources/utils/importMNISTDatasets.cpp > CMakeFiles/utils.dir/importMNISTDatasets.cpp.i

sources/utils/CMakeFiles/utils.dir/importMNISTDatasets.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/utils.dir/importMNISTDatasets.cpp.s"
	cd /home/dante/dev/NilDa/Release/sources/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dante/dev/NilDa/sources/utils/importMNISTDatasets.cpp -o CMakeFiles/utils.dir/importMNISTDatasets.cpp.s

sources/utils/CMakeFiles/utils.dir/images.cpp.o: sources/utils/CMakeFiles/utils.dir/flags.make
sources/utils/CMakeFiles/utils.dir/images.cpp.o: ../sources/utils/images.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object sources/utils/CMakeFiles/utils.dir/images.cpp.o"
	cd /home/dante/dev/NilDa/Release/sources/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/utils.dir/images.cpp.o -c /home/dante/dev/NilDa/sources/utils/images.cpp

sources/utils/CMakeFiles/utils.dir/images.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/utils.dir/images.cpp.i"
	cd /home/dante/dev/NilDa/Release/sources/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dante/dev/NilDa/sources/utils/images.cpp > CMakeFiles/utils.dir/images.cpp.i

sources/utils/CMakeFiles/utils.dir/images.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/utils.dir/images.cpp.s"
	cd /home/dante/dev/NilDa/Release/sources/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dante/dev/NilDa/sources/utils/images.cpp -o CMakeFiles/utils.dir/images.cpp.s

# Object files for target utils
utils_OBJECTS = \
"CMakeFiles/utils.dir/Random.cpp.o" \
"CMakeFiles/utils.dir/importMNISTDatasets.cpp.o" \
"CMakeFiles/utils.dir/images.cpp.o"

# External object files for target utils
utils_EXTERNAL_OBJECTS =

sources/utils/libutils.a: sources/utils/CMakeFiles/utils.dir/Random.cpp.o
sources/utils/libutils.a: sources/utils/CMakeFiles/utils.dir/importMNISTDatasets.cpp.o
sources/utils/libutils.a: sources/utils/CMakeFiles/utils.dir/images.cpp.o
sources/utils/libutils.a: sources/utils/CMakeFiles/utils.dir/build.make
sources/utils/libutils.a: sources/utils/CMakeFiles/utils.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dante/dev/NilDa/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libutils.a"
	cd /home/dante/dev/NilDa/Release/sources/utils && $(CMAKE_COMMAND) -P CMakeFiles/utils.dir/cmake_clean_target.cmake
	cd /home/dante/dev/NilDa/Release/sources/utils && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/utils.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sources/utils/CMakeFiles/utils.dir/build: sources/utils/libutils.a

.PHONY : sources/utils/CMakeFiles/utils.dir/build

sources/utils/CMakeFiles/utils.dir/clean:
	cd /home/dante/dev/NilDa/Release/sources/utils && $(CMAKE_COMMAND) -P CMakeFiles/utils.dir/cmake_clean.cmake
.PHONY : sources/utils/CMakeFiles/utils.dir/clean

sources/utils/CMakeFiles/utils.dir/depend:
	cd /home/dante/dev/NilDa/Release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dante/dev/NilDa /home/dante/dev/NilDa/sources/utils /home/dante/dev/NilDa/Release /home/dante/dev/NilDa/Release/sources/utils /home/dante/dev/NilDa/Release/sources/utils/CMakeFiles/utils.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sources/utils/CMakeFiles/utils.dir/depend

