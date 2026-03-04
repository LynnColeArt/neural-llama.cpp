ggml_add_backend_library(ggml-coreml
                         ggml-coreml.cpp
                        )
if (TARGET ggml-metal)
    target_link_libraries(ggml-coreml PRIVATE ggml-metal)
    target_compile_definitions(ggml-coreml PRIVATE GGML_USE_METAL)
endif()
