#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// CoreML backend API

GGML_BACKEND_API ggml_backend_t ggml_backend_coreml_init(void);

GGML_BACKEND_API bool ggml_backend_is_coreml(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_coreml_reg(void);

#ifdef __cplusplus
}
#endif
