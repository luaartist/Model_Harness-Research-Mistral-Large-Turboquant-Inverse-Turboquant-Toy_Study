#include "ggml_tq_tensor_shim.h"
#include "ggml_tq_http_bridge.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int set_error(char *err, size_t err_cap, const char *fmt, ...) {
    if (err != NULL && err_cap > 0) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(err, err_cap, fmt, args);
        va_end(args);
    }
    return -1;
}

static int checked_mul(size_t lhs, size_t rhs, size_t *out) {
    if (lhs != 0 && rhs > SIZE_MAX / lhs) {
        return -1;
    }
    *out = lhs * rhs;
    return 0;
}

const char *tq_runner_backend_name(tq_runner_backend_kind backend) {
    switch (backend) {
    case TQ_RUNNER_BACKEND_DECODED_F32_FILE:
        return "decoded_f32_file";
    case TQ_RUNNER_BACKEND_HTTP_BRIDGE:
        return "http_bridge";
    case TQ_RUNNER_BACKEND_INPROCESS_OPENCL:
        return "inprocess_opencl";
    default:
        return "unknown";
    }
}

int tq_runner_backend_from_string(
    const char *value,
    tq_runner_backend_kind *out,
    char *err,
    size_t err_cap
) {
    const char *backend = value != NULL && value[0] != '\0' ? value : "decoded_f32_file";

    if (out == NULL) {
        return set_error(err, err_cap, "backend output is null");
    }
    if (strcmp(backend, "decoded_f32_file") == 0 ||
        strcmp(backend, "decoded-f32-file") == 0 ||
        strcmp(backend, "file") == 0) {
        *out = TQ_RUNNER_BACKEND_DECODED_F32_FILE;
        return 0;
    }
    if (strcmp(backend, "http_bridge") == 0 || strcmp(backend, "http-bridge") == 0) {
        *out = TQ_RUNNER_BACKEND_HTTP_BRIDGE;
        return 0;
    }
    if (strcmp(backend, "inprocess_opencl") == 0 ||
        strcmp(backend, "inprocess-opencl") == 0) {
        *out = TQ_RUNNER_BACKEND_INPROCESS_OPENCL;
        return 0;
    }
    return set_error(err, err_cap, "unknown backend: %s", backend);
}

int tq_tensor_desc_validate(const tq_tensor_desc *desc, char *err, size_t err_cap) {
    size_t row_bytes = 0;
    size_t expected_bytes = 0;

    if (desc == NULL) {
        return set_error(err, err_cap, "descriptor is null");
    }
    if (desc->path == NULL || desc->path[0] == '\0') {
        return set_error(err, err_cap, "descriptor path is empty");
    }
    if (desc->rows == 0 || desc->cols == 0) {
        return set_error(err, err_cap, "rows and cols must be nonzero");
    }
    if (desc->item_size != sizeof(float)) {
        return set_error(err, err_cap, "item_size must be 4 for GGML_TYPE_F32");
    }
    if (checked_mul(desc->cols, desc->item_size, &row_bytes) != 0) {
        return set_error(err, err_cap, "row byte calculation overflowed");
    }
    if (checked_mul(desc->rows, row_bytes, &expected_bytes) != 0) {
        return set_error(err, err_cap, "tensor byte calculation overflowed");
    }
    if (desc->nbytes != expected_bytes) {
        return set_error(
            err,
            err_cap,
            "nbytes mismatch: got %zu expected %zu",
            desc->nbytes,
            expected_bytes
        );
    }
    if (desc->ne[0] != desc->cols || desc->ne[1] != desc->rows ||
        desc->ne[2] != 1 || desc->ne[3] != 1) {
        return set_error(err, err_cap, "GGML ne[] does not match row-major shape");
    }
    if (desc->nb[0] != desc->item_size || desc->nb[1] != row_bytes ||
        desc->nb[2] != desc->nbytes || desc->nb[3] != desc->nbytes) {
        return set_error(err, err_cap, "GGML nb[] does not match contiguous f32 layout");
    }
    return 0;
}

int tq_load_f32_tensor(
    const tq_tensor_desc *desc,
    tq_tensor_buffer *out,
    char *err,
    size_t err_cap
) {
    FILE *handle = NULL;
    long file_size = 0;
    size_t read_bytes = 0;
    size_t value_count = 0;
    size_t sample_len = 0;
    float *values = NULL;
    double sum_sq = 0.0;

    if (tq_tensor_desc_validate(desc, err, err_cap) != 0) {
        return -1;
    }
    if (out == NULL) {
        return set_error(err, err_cap, "output buffer is null");
    }
    memset(out, 0, sizeof(*out));

    handle = fopen(desc->path, "rb");
    if (handle == NULL) {
        return set_error(err, err_cap, "failed to open %s: %s", desc->path, strerror(errno));
    }
    if (fseek(handle, 0, SEEK_END) != 0) {
        fclose(handle);
        return set_error(err, err_cap, "failed to seek %s", desc->path);
    }
    file_size = ftell(handle);
    if (file_size < 0) {
        fclose(handle);
        return set_error(err, err_cap, "failed to size %s", desc->path);
    }
    if ((size_t)file_size != desc->nbytes) {
        fclose(handle);
        return set_error(
            err,
            err_cap,
            "file size mismatch: got %ld expected %zu",
            file_size,
            desc->nbytes
        );
    }
    if (fseek(handle, 0, SEEK_SET) != 0) {
        fclose(handle);
        return set_error(err, err_cap, "failed to rewind %s", desc->path);
    }

    values = (float *)malloc(desc->nbytes);
    if (values == NULL) {
        fclose(handle);
        return set_error(err, err_cap, "failed to allocate %zu bytes", desc->nbytes);
    }
    read_bytes = fread(values, 1, desc->nbytes, handle);
    fclose(handle);
    if (read_bytes != desc->nbytes) {
        free(values);
        return set_error(err, err_cap, "short read: got %zu expected %zu", read_bytes, desc->nbytes);
    }

    value_count = desc->nbytes / sizeof(float);
    sample_len = value_count < 8 ? value_count : 8;
    for (size_t index = 0; index < value_count; ++index) {
        const double value = (double)values[index];
        if (!isfinite(value)) {
            free(values);
            return set_error(err, err_cap, "non-finite value at index %zu", index);
        }
        sum_sq += value * value;
        if (index < sample_len) {
            out->sample[index] = values[index];
        }
    }

    out->desc = *desc;
    out->data = values;
    out->nbytes = desc->nbytes;
    out->l2_norm = sqrt(sum_sq);
    out->sample_len = sample_len;
    return 0;
}

static void write_json_string(FILE *handle, const char *value) {
    const unsigned char *cursor = (const unsigned char *)(value != NULL ? value : "");
    fputc('"', handle);
    while (*cursor != '\0') {
        if (*cursor == '"' || *cursor == '\\') {
            fputc('\\', handle);
            fputc((int)*cursor, handle);
        } else if (*cursor >= 0x20 && *cursor < 0x7f) {
            fputc((int)*cursor, handle);
        } else {
            fprintf(handle, "\\u%04x", (unsigned int)*cursor);
        }
        ++cursor;
    }
    fputc('"', handle);
}

int tq_write_audit_json(
    const tq_tensor_buffer *buffer,
    const char *audit_path,
    char *err,
    size_t err_cap
) {
    FILE *handle = NULL;

    if (buffer == NULL || buffer->data == NULL) {
        return set_error(err, err_cap, "buffer is empty");
    }
    if (audit_path == NULL || audit_path[0] == '\0') {
        return set_error(err, err_cap, "audit path is empty");
    }

    handle = fopen(audit_path, "wb");
    if (handle == NULL) {
        return set_error(err, err_cap, "failed to open audit %s: %s", audit_path, strerror(errno));
    }

    fprintf(handle, "{\n");
    fprintf(handle, "  \"ok\": true,\n");
    fprintf(handle, "  \"adapter\": \"private-runner-c-shim\",\n");
    fprintf(handle, "  \"tensor\": {\n");
    fprintf(handle, "    \"name\": ");
    write_json_string(handle, buffer->desc.name);
    fprintf(handle, ",\n");
    fprintf(handle, "    \"path\": ");
    write_json_string(handle, buffer->desc.path);
    fprintf(handle, ",\n");
    fprintf(handle, "    \"ggml_type\": \"GGML_TYPE_F32\",\n");
    fprintf(handle, "    \"rows\": %zu,\n", buffer->desc.rows);
    fprintf(handle, "    \"cols\": %zu,\n", buffer->desc.cols);
    fprintf(
        handle,
        "    \"ggml_ne\": [%zu, %zu, %zu, %zu],\n",
        buffer->desc.ne[0],
        buffer->desc.ne[1],
        buffer->desc.ne[2],
        buffer->desc.ne[3]
    );
    fprintf(
        handle,
        "    \"ggml_nb\": [%zu, %zu, %zu, %zu],\n",
        buffer->desc.nb[0],
        buffer->desc.nb[1],
        buffer->desc.nb[2],
        buffer->desc.nb[3]
    );
    fprintf(handle, "    \"nbytes\": %zu,\n", buffer->nbytes);
    fprintf(handle, "    \"l2_norm\": %.17g,\n", buffer->l2_norm);
    fprintf(handle, "    \"expected_sha256\": ");
    write_json_string(handle, buffer->desc.expected_sha256);
    fprintf(handle, "\n  },\n");
    fprintf(handle, "  \"sample\": [");
    for (size_t index = 0; index < buffer->sample_len; ++index) {
        fprintf(handle, "%s%.9g", index == 0 ? "" : ", ", (double)buffer->sample[index]);
    }
    fprintf(handle, "],\n");
    fprintf(handle, "  \"loader_contract\": {\n");
    fprintf(handle, "    \"allocation\": \"ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows)\",\n");
    fprintf(handle, "    \"copy\": \"memcpy(tensor->data, buffer.data, buffer.nbytes)\",\n");
    fprintf(handle, "    \"transpose_required\": false\n");
    fprintf(handle, "  }\n");
    fprintf(handle, "}\n");
    fclose(handle);
    return 0;
}

int tq_write_runner_audit_json(
    const tq_runner_request *request,
    const tq_runner_result *result,
    const char *audit_path,
    char *err,
    size_t err_cap
) {
    FILE *handle = NULL;
    const tq_tensor_buffer *buffer = result != NULL ? &result->tensor : NULL;

    if (request == NULL) {
        return set_error(err, err_cap, "runner request is null");
    }
    if (result == NULL || buffer == NULL || buffer->data == NULL) {
        return set_error(err, err_cap, "runner result is empty");
    }
    if (audit_path == NULL || audit_path[0] == '\0') {
        return set_error(err, err_cap, "audit path is empty");
    }

    handle = fopen(audit_path, "wb");
    if (handle == NULL) {
        return set_error(err, err_cap, "failed to open audit %s: %s", audit_path, strerror(errno));
    }

    fprintf(handle, "{\n");
    fprintf(handle, "  \"ok\": true,\n");
    fprintf(handle, "  \"adapter\": \"private-runner-api\",\n");
    fprintf(handle, "  \"backend\": {\n");
    fprintf(handle, "    \"kind\": ");
    write_json_string(handle, result->backend_name);
    fprintf(handle, ",\n");
    fprintf(
        handle,
        "    \"used_predecoded_file\": %s,\n",
        result->used_predecoded_file ? "true" : "false"
    );
    fprintf(handle, "    \"url\": ");
    write_json_string(handle, request->backend_url);
    fprintf(handle, ",\n");
    fprintf(handle, "    \"used_http_bridge\": %s,\n", result->used_http_bridge ? "true" : "false");
    fprintf(
        handle,
        "    \"implemented\": %s\n",
        (result->used_predecoded_file || result->used_http_bridge) ? "true" : "false"
    );
    fprintf(handle, "  },\n");
    fprintf(handle, "  \"package\": {\n");
    fprintf(handle, "    \"manifest\": ");
    write_json_string(handle, request->package_manifest_path);
    fprintf(handle, ",\n");
    fprintf(handle, "    \"sidecar_metadata\": ");
    write_json_string(handle, request->sidecar_metadata_path);
    fprintf(handle, "\n  },\n");
    fprintf(handle, "  \"tensor\": {\n");
    fprintf(handle, "    \"name\": ");
    write_json_string(handle, buffer->desc.name);
    fprintf(handle, ",\n");
    fprintf(handle, "    \"path\": ");
    write_json_string(handle, buffer->desc.path);
    fprintf(handle, ",\n");
    fprintf(handle, "    \"ggml_type\": \"GGML_TYPE_F32\",\n");
    fprintf(handle, "    \"rows\": %zu,\n", buffer->desc.rows);
    fprintf(handle, "    \"cols\": %zu,\n", buffer->desc.cols);
    fprintf(
        handle,
        "    \"ggml_ne\": [%zu, %zu, %zu, %zu],\n",
        buffer->desc.ne[0],
        buffer->desc.ne[1],
        buffer->desc.ne[2],
        buffer->desc.ne[3]
    );
    fprintf(
        handle,
        "    \"ggml_nb\": [%zu, %zu, %zu, %zu],\n",
        buffer->desc.nb[0],
        buffer->desc.nb[1],
        buffer->desc.nb[2],
        buffer->desc.nb[3]
    );
    fprintf(handle, "    \"nbytes\": %zu,\n", buffer->nbytes);
    fprintf(handle, "    \"l2_norm\": %.17g,\n", buffer->l2_norm);
    fprintf(handle, "    \"expected_sha256\": ");
    write_json_string(handle, buffer->desc.expected_sha256);
    fprintf(handle, "\n  },\n");
    fprintf(handle, "  \"sample\": [");
    for (size_t index = 0; index < buffer->sample_len; ++index) {
        fprintf(handle, "%s%.9g", index == 0 ? "" : ", ", (double)buffer->sample[index]);
    }
    fprintf(handle, "],\n");
    fprintf(handle, "  \"loader_contract\": {\n");
    fprintf(handle, "    \"allocation\": \"ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows)\",\n");
    fprintf(handle, "    \"copy\": \"memcpy(tensor->data, result.tensor.data, result.tensor.nbytes)\",\n");
    fprintf(handle, "    \"transpose_required\": false\n");
    fprintf(handle, "  }\n");
    fprintf(handle, "}\n");
    fclose(handle);
    return 0;
}

int tq_runner_decode(
    const tq_runner_request *request,
    tq_runner_result *result,
    char *err,
    size_t err_cap
) {
    if (request == NULL) {
        return set_error(err, err_cap, "runner request is null");
    }
    if (result == NULL) {
        return set_error(err, err_cap, "runner result is null");
    }
    memset(result, 0, sizeof(*result));
    result->backend = request->backend;
    result->backend_name = tq_runner_backend_name(request->backend);

    switch (request->backend) {
    case TQ_RUNNER_BACKEND_DECODED_F32_FILE:
        if (tq_load_f32_tensor(&request->tensor, &result->tensor, err, err_cap) != 0) {
            return -1;
        }
        result->used_predecoded_file = 1;
        break;
    case TQ_RUNNER_BACKEND_HTTP_BRIDGE:
        if (tq_http_bridge_decode(request, result, err, err_cap) != 0) {
            return -1;
        }
        break;
    case TQ_RUNNER_BACKEND_INPROCESS_OPENCL:
        return set_error(err, err_cap, "inprocess_opencl backend is not implemented yet");
    default:
        return set_error(err, err_cap, "unknown runner backend");
    }

    if (request->audit_path != NULL && request->audit_path[0] != '\0') {
        if (tq_write_runner_audit_json(request, result, request->audit_path, err, err_cap) != 0) {
            tq_free_tensor(&result->tensor);
            memset(result, 0, sizeof(*result));
            return -1;
        }
        result->audit_written = 1;
    }

    return 0;
}

void tq_free_tensor(tq_tensor_buffer *buffer) {
    if (buffer != NULL) {
        if (buffer->owns_desc_strings) {
            free((void *)buffer->desc.name);
            free((void *)buffer->desc.path);
            free((void *)buffer->desc.expected_sha256);
        }
        free(buffer->data);
        memset(buffer, 0, sizeof(*buffer));
    }
}

void tq_runner_result_free(tq_runner_result *result) {
    if (result != NULL) {
        tq_free_tensor(&result->tensor);
        memset(result, 0, sizeof(*result));
    }
}