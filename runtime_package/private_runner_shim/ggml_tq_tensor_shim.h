#ifndef GGML_TQ_TENSOR_SHIM_H
#define GGML_TQ_TENSOR_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tq_tensor_desc {
    const char *name;
    const char *path;
    const char *expected_sha256;
    size_t rows;
    size_t cols;
    size_t item_size;
    size_t ne[4];
    size_t nb[4];
    size_t nbytes;
} tq_tensor_desc;

typedef struct tq_tensor_buffer {
    tq_tensor_desc desc;
    void *data;
    size_t nbytes;
    double l2_norm;
    float sample[8];
    size_t sample_len;
    int owns_desc_strings;
} tq_tensor_buffer;

typedef enum tq_runner_backend_kind {
    TQ_RUNNER_BACKEND_DECODED_F32_FILE = 0,
    TQ_RUNNER_BACKEND_HTTP_BRIDGE = 1,
    TQ_RUNNER_BACKEND_INPROCESS_OPENCL = 2
} tq_runner_backend_kind;

typedef struct tq_runner_request {
    tq_runner_backend_kind backend;
    tq_tensor_desc tensor;
    const char *audit_path;
    const char *package_manifest_path;
    const char *sidecar_metadata_path;
    const char *backend_url;
} tq_runner_request;

typedef struct tq_runner_result {
    tq_runner_backend_kind backend;
    const char *backend_name;
    int used_predecoded_file;
    int used_http_bridge;
    int audit_written;
    tq_tensor_buffer tensor;
} tq_runner_result;

const char *tq_runner_backend_name(tq_runner_backend_kind backend);

int tq_runner_backend_from_string(
    const char *value,
    tq_runner_backend_kind *out,
    char *err,
    size_t err_cap
);

int tq_tensor_desc_validate(const tq_tensor_desc *desc, char *err, size_t err_cap);

int tq_load_f32_tensor(
    const tq_tensor_desc *desc,
    tq_tensor_buffer *out,
    char *err,
    size_t err_cap
);

int tq_write_audit_json(
    const tq_tensor_buffer *buffer,
    const char *audit_path,
    char *err,
    size_t err_cap
);

int tq_runner_decode(
    const tq_runner_request *request,
    tq_runner_result *result,
    char *err,
    size_t err_cap
);

int tq_write_runner_audit_json(
    const tq_runner_request *request,
    const tq_runner_result *result,
    const char *audit_path,
    char *err,
    size_t err_cap
);

void tq_free_tensor(tq_tensor_buffer *buffer);

void tq_runner_result_free(tq_runner_result *result);

#ifdef __cplusplus
}
#endif

#endif