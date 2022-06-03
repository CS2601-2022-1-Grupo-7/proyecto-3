# Proyecto-3
# Copyright © 2022 Grupo 7
#
# proyecto-3 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# proyecto-3 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hello.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.13.0)

project(proyecto-3
	VERSION 0.0.0
	DESCRIPTION "Proyecto 3"
	HOMEPAGE_URL "https://github.com/CS2601-2022-1-Grupo-7/proyecto-3"
)

# Without this, paths are not relative in the sources list
cmake_policy(SET CMP0076 NEW)

include(GNUInstallDirs)
include(FindPkgConfig)
include(CPack)

# The program itself
add_executable(${PROJECT_NAME})

# C++ version
set_target_properties(${PROJECT_NAME}
	PROPERTIES
		CXX_STANDARD 20
)

add_subdirectory(src) # Sources list
add_subdirectory(pkg) # Packaging

# Libraries
pkg_check_modules(libraries
	REQUIRED IMPORTED_TARGET
		RapidJSON
		eigen3
)

# Linking
target_link_libraries(${PROJECT_NAME}
	PRIVATE
		PkgConfig::libraries
)

# Default flags
if(UNIX)
	if(NOT (DEFINED ENV{CFLAGS} OR CMAKE_C_FLAGS))
		set(CMAKE_C_FLAGS "-Wall -Wextra -g -Og")
	endif()
	if(NOT (DEFINED ENV{CXXFLAGS} OR CMAKE_CXX_FLAGS))
		set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -Og")
	endif()
endif()

# Install target
install(TARGETS ${PROJECT_NAME}
	DESTINATION ${CMAKE_INSTALL_BINDIR}
)
