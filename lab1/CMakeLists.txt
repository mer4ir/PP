cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME matrix_mult)

set(SOURCES
    matrix_mult.cpp
)

add_executable(${TARGET_NAME} ${SOURCES})

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

add_custom_target(run
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}
    DEPENDS ${TARGET_NAME}
    COMMENT "Запуск программы..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(verify
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} -r 5 results.txt
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_results.py 
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_A.txt 
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_B.txt 
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_C.txt
    DEPENDS ${TARGET_NAME}
    COMMENT "Запуск с верификацией..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)