cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
project(DiscreteAIEM_cuda VERSION 1.0 LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

file(GLOB SOURCES_FILES 
    "./*.cu" 
    "./*.h"
)

add_executable(${PROJECT_NAME} ${SOURCES_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC 
    "./"
)


set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${CUDA_LIBRARIES}
)



