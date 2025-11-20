
add_executable(algorithm.test algorithm.test.cpp)
target_link_libraries(algorithm.test PUBLIC codeGen)

add_executable(execution_validation.test execution_validation.test.cpp)
target_link_libraries(execution_validation.test PUBLIC codeGen)

add_executable(model_loading.test model_loading.test.cpp)
target_link_libraries(model_loading.test PUBLIC codeGen)

