#include "ggml_tq_loader_adapter.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
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

int tq_ggml_context_bytes(
    const tq_tensor_buffer *buffer,
    size_t *out,
    char *err,
    size_t err_cap
) {
    const size_t scratch_bytes = 1024 * 1024;
    if (buffer == NULL || buffer->data == NULL) {
        return set_error(err, err_cap, "source tensor buffer is empty");
    }
    if (out == NULL) {
        return set_error(err, err_cap, "context byte output is null");
    }
    if (buffer->nbytes > SIZE_MAX - scratch_bytes - ggml_tensor_overhead()) {
        return set_error(err, err_cap, "GGML context byte calculation overflowed");
    }
    *out = buffer->nbytes + ggml_tensor_overhead() + scratch_bytes;
    return 0;
}

int tq_ggml_materialize_f32(
    struct ggml_context *ctx,
    const tq_tensor_buffer *buffer,
    tq_ggml_loader_tensor *out,
    char *err,
    size_t err_cap
) {
    struct ggml_tensor *tensor = NULL;
    char desc_err[256];

    if (ctx == NULL) {
        return set_error(err, err_cap, "GGML context is null");
    }
    if (buffer == NULL || buffer->data == NULL) {
        return set_error(err, err_cap, "source tensor buffer is empty");
    }
    if (out == NULL) {
        return set_error(err, err_cap, "loader tensor output is null");
    }
    if (tq_tensor_desc_validate(&buffer->desc, desc_err, sizeof(desc_err)) != 0) {
        return set_error(err, err_cap, "source descriptor invalid: %s", desc_err);
    }
    if (buffer->desc.cols > (size_t)INT64_MAX || buffer->desc.rows > (size_t)INT64_MAX) {
        return set_error(err, err_cap, "source shape exceeds GGML int64 dimensions");
    }

    memset(out, 0, sizeof(*out));
    tensor = ggml_new_tensor_2d(
        ctx,
        GGML_TYPE_F32,
        (int64_t)buffer->desc.cols,
        (int64_t)buffer->desc.rows
    );
    if (tensor == NULL) {
        return set_error(err, err_cap, "ggml_new_tensor_2d returned null");
    }
    ggml_set_name(tensor, buffer->desc.name);

    if (tensor->data == NULL) {
        return set_error(err, err_cap, "GGML tensor data pointer is null");
    }
    if (tensor->type != GGML_TYPE_F32) {
        return set_error(err, err_cap, "GGML tensor type is not F32");
    }
    if ((size_t)tensor->ne[0] != buffer->desc.cols ||
        (size_t)tensor->ne[1] != buffer->desc.rows ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        return set_error(err, err_cap, "GGML tensor shape does not match source descriptor");
    }
    if (tensor->nb[0] != buffer->desc.nb[0] ||
        tensor->nb[1] != buffer->desc.nb[1] ||
        tensor->nb[2] != buffer->desc.nb[2] ||
        tensor->nb[3] != buffer->desc.nb[3]) {
        return set_error(err, err_cap, "GGML tensor strides do not match source descriptor");
    }
    if (ggml_nbytes(tensor) != buffer->nbytes) {
        return set_error(
            err,
            err_cap,
            "GGML tensor byte count mismatch: got %zu expected %zu",
            ggml_nbytes(tensor),
            buffer->nbytes
        );
    }
    if (!ggml_is_contiguous(tensor)) {
        return set_error(err, err_cap, "GGML tensor is not contiguous");
    }

    memcpy(tensor->data, buffer->data, buffer->nbytes);
    if (memcmp(tensor->data, buffer->data, buffer->nbytes) != 0) {
        return set_error(err, err_cap, "GGML tensor copy did not preserve byte identity");
    }

    out->tensor = tensor;
    out->nbytes = ggml_nbytes(tensor);
    out->copied_bytes = buffer->nbytes;
    out->byte_identity_ok = 1;
    out->contiguous_ok = 1;
    return 0;
}

int tq_ggml_write_loader_audit_json(
    const tq_runner_result *runner,
    const tq_ggml_loader_tensor *loaded,
    const char *audit_path,
    char *err,
    size_t err_cap
) {
    FILE *handle = NULL;
    const struct ggml_tensor *tensor = loaded != NULL ? loaded->tensor : NULL;

    if (runner == NULL || runner->tensor.data == NULL) {
        return set_error(err, err_cap, "runner result is empty");
    }
    if (loaded == NULL || tensor == NULL) {
        return set_error(err, err_cap, "GGML loader tensor is empty");
    }
    if (audit_path == NULL || audit_path[0] == '\0') {
        return set_error(err, err_cap, "loader audit path is empty");
    }

    handle = fopen(audit_path, "wb");
    if (handle == NULL) {
        return set_error(err, err_cap, "failed to open loader audit %s: %s", audit_path, strerror(errno));
    }

    fprintf(handle, "{\n");
    fprintf(handle, "  \"ok\": true,\n");
    fprintf(handle, "  \"adapter\": \"ggml-loader-surface-smoke\",\n");
    fprintf(handle, "  \"source_backend\": ");
    write_json_string(handle, runner->backend_name);
    fprintf(handle, ",\n");
    fprintf(handle, "  \"ggml\": {\n");
    fprintf(handle, "    \"version\": ");
    write_json_string(handle, ggml_version());
    fprintf(handle, ",\n");
    fprintf(handle, "    \"type\": ");
    write_json_string(handle, ggml_type_name(tensor->type));
    fprintf(handle, ",\n");
    fprintf(handle, "    \"name\": ");
    write_json_string(handle, ggml_get_name(tensor));
    fprintf(handle, ",\n");
    fprintf(
        handle,
        "    \"ne\": [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "],\n",
        tensor->ne[0],
        tensor->ne[1],
        tensor->ne[2],
        tensor->ne[3]
    );
    fprintf(
        handle,
        "    \"nb\": [%zu, %zu, %zu, %zu],\n",
        tensor->nb[0],
        tensor->nb[1],
        tensor->nb[2],
        tensor->nb[3]
    );
    fprintf(handle, "    \"nbytes\": %zu,\n", ggml_nbytes(tensor));
    fprintf(handle, "    \"contiguous\": %s\n", ggml_is_contiguous(tensor) ? "true" : "false");
    fprintf(handle, "  },\n");
    fprintf(handle, "  \"source_tensor\": {\n");
    fprintf(handle, "    \"name\": ");
    write_json_string(handle, runner->tensor.desc.name);
    fprintf(handle, ",\n");
    fprintf(handle, "    \"rows\": %zu,\n", runner->tensor.desc.rows);
    fprintf(handle, "    \"cols\": %zu,\n", runner->tensor.desc.cols);
    fprintf(handle, "    \"nbytes\": %zu,\n", runner->tensor.nbytes);
    fprintf(handle, "    \"l2_norm\": %.17g\n", runner->tensor.l2_norm);
    fprintf(handle, "  },\n");
    fprintf(handle, "  \"copy_check\": {\n");
    fprintf(handle, "    \"copied_bytes\": %zu,\n", loaded->copied_bytes);
    fprintf(handle, "    \"byte_identity_ok\": %s,\n", loaded->byte_identity_ok ? "true" : "false");
    fprintf(handle, "    \"transpose_required\": false\n");
    fprintf(handle, "  },\n");
    fprintf(handle, "  \"loader_contract\": {\n");
    fprintf(handle, "    \"allocation\": \"ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows)\",\n");
    fprintf(handle, "    \"copy\": \"memcpy(ggml_tensor->data, runner.tensor.data, runner.tensor.nbytes)\",\n");
    fprintf(handle, "    \"validated_against_real_ggml\": true\n");
    fprintf(handle, "  },\n");
    fprintf(handle, "  \"sample\": [");
    for (size_t index = 0; index < runner->tensor.sample_len; ++index) {
        fprintf(handle, "%s%.9g", index == 0 ? "" : ", ", (double)runner->tensor.sample[index]);
    }
    fprintf(handle, "]\n");
    fprintf(handle, "}\n");
    fclose(handle);
    return 0;
}