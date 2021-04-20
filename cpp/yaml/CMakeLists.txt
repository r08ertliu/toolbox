cmake_minimum_required(VERSION 3.2)

project(parse_yaml CXX)

include_directories(yaml-cpp/include)

add_subdirectory(yaml-cpp)
add_executable(parse_yaml parse_yaml.cpp)
target_link_libraries(parse_yaml yaml-cpp)
