# ============================================================
# TitanInfer Example Targets
# ============================================================
add_executable(simple_inference simple_inference.cpp)
target_link_libraries(simple_inference PRIVATE titaninfer)

add_executable(batch_processing batch_processing.cpp)
target_link_libraries(batch_processing PRIVATE titaninfer)

add_executable(c_api_usage c_api_usage.c)
target_link_libraries(c_api_usage PRIVATE titaninfer)
