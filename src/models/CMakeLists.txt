
add_library(models SHARED 
            base1d.cpp 
            base2d.cpp 
            base3d.cpp 
            interpolation.cpp 
            model_common.cpp 
            model1d.cpp 
            model2d.cpp 
            model3d.cpp
            ../utils/common.cpp)

target_compile_options(models PUBLIC ${comp_flags})
  