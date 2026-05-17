#ifndef GGML_TQ_LOADER_ADAPTER_H
#define GGML_TQ_LOADER_ADAPTER_H

#include "ggml_tq_tensor_shim.h"

#include <ggml.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tq_ggml_loader_tensor {
    struct ggml_tensor *tensor;
    size_t nbytes;
    size_t copied_bytes;
    int byte_identity_ok;
    int contiguous_ok;
} tq_ggml_loader_tensor;

int tq_ggml_context_bytes(
    const tq_tensor_buffer *buffer,
    size_t *out,
    char *err,
    size_t err_cap
);

int tq_ggml_materialize_f32(
    struct ggml_context *ctx,
    const tq_tensor_buffer *buffer,
    tq_ggml_loader_tensor *out,
    char *err,
    size_t err_cap
);

int tq_ggml_write_loader_audit_json(
    const tq_runner_result *runner,
    const tq_ggml_loader_tensor *loaded,
    const char *audit_path,
    char *err,
    size_t err_cap
);

#ifdef __cplusplus
}
#endif

#endif