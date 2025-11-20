add_executable(experiment1 experiment1.cpp)
target_link_libraries(experiment1 PUBLIC codeGen)

add_executable(experiment2 experiment2.cpp)
target_link_libraries(experiment2 PUBLIC codeGen)

add_executable(generate_chains_10comb generate_chains_10comb.cpp)
target_link_libraries(generate_chains_10comb PUBLIC codeGen)

add_executable(gen_armadillo_code gen_armadillo_code.cpp)
target_link_libraries(gen_armadillo_code PUBLIC codeGen)

add_executable(timings_10comb timings_10comb.cpp)
target_link_libraries(timings_10comb PUBLIC codeGen)

add_executable(generate_models generate_models.cpp)
target_link_libraries(generate_models PUBLIC codeGen)

add_custom_command(
    OUTPUT ${TMP_DIR}/arma_code.cpp
    COMMAND gen_armadillo_code ${TMP_DIR}/expr_exp2.txt 1000 ${TMP_DIR}/
    DEPENDS gen_armadillo_code
    COMMENT "Generating Armadillo code in ${TMP_DIR}"
)
    
set(PATH_ARMA_CODE "${TMP_DIR}/arma_code.hpp")
configure_file(experiment_arma.hpp.in ${TMP_DIR}/experiment_arma.hpp @ONLY)

add_executable(experiment_arma experiment_arma.cpp ${TMP_DIR}/arma_code.cpp)
target_include_directories(experiment_arma PUBLIC "${TMP_DIR}")
target_link_libraries(experiment_arma PUBLIC codeGen ${ARMADILLO_LIBRARY})