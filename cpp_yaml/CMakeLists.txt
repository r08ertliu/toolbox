cmake_minimum_required(VERSION 3.2)

project(cpp_yaml CXX)

include_directories(yaml-cpp/include)

add_subdirectory(yaml-cpp)
add_executable(cpp_yaml cpp_yaml.cpp)
#target_link_libraries(yaml-cpp ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp/libyaml-cpp.a)
target_link_libraries(cpp_yaml yaml-cpp)
