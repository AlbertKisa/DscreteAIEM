# 设置CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(aiem VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加可执行文件
add_executable(aiem
    main.cpp
    complex0.cpp
    Fun.cpp
    math_product.cpp
)

# 包含头文件目录
target_include_directories(aiem PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 如果有其他库需要链接，可以在这里添加
# target_link_libraries(MyProject PRIVATE some_library)
