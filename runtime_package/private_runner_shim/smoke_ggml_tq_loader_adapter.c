#include "ggml_tq_loader_adapter.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>

int main(int argc, char **argv) {
    char err[512];
    tq_runner_backend_kind backend;
    tq_runner_result result;
    tq_ggml_loader_tensor loaded;
    struct ggml_context *ctx = NULL;
    size_t ctx_bytes = 0;
    const char *input_path = argc > 1 ? argv[1] : "../../results/ggml_opencl_sidecar_decode.f32";
    const char *audit_path = argc > 2 ? argv[2] : "../../results/ggml_loader_surface_audit.json";
    const char *backend_arg = argc > 3 ? argv[3] : "decoded_f32_file";
    const char *manifest_path = argc > 4 ? argv[4] : "../runtime_package_manifest.json";
    const char *backend_url = argc > 5 ? argv[5] : "http://127.0.0.1:8504";
    const tq_tensor_desc desc = {
        .name = "c0.q4.flavor_best.decoded_f32",
        .path = input_path,
        .expected_sha256 = "0213eed45ca2b664f4979228ffdb332eb0624734daf717012016df6c76d1a2a8",
        .rows = 256,
        .cols = 128,
        .item_size = 4,
        .ne = {128, 256, 1, 1},
        .nb = {4, 512, 131072, 131072},
        .nbytes = 131072,
    };
    const tq_runner_request request = {
        .backend = TQ_RUNNER_BACKEND_DECODED_F32_FILE,
        .tensor = desc,
        .audit_path = "",
        .package_manifest_path = manifest_path,
        .sidecar_metadata_path = "../../results/ggml_opencl_sidecar_decode.json",
        .backend_url = backend_url,
    };
    tq_runner_request selected_request = request;

    if (tq_runner_backend_from_string(backend_arg, &backend, err, sizeof(err)) != 0) {
        fprintf(stderr, "backend failed: %s\n", err);
        return 1;
    }
    selected_request.backend = backend;

    if (tq_runner_decode(&selected_request, &result, err, sizeof(err)) != 0) {
        fprintf(stderr, "runner failed: %s\n", err);
        return 1;
    }

    if (tq_ggml_context_bytes(&result.tensor, &ctx_bytes, err, sizeof(err)) != 0) {
        fprintf(stderr, "context sizing failed: %s\n", err);
        tq_runner_result_free(&result);
        return 1;
    }

    const struct ggml_init_params params = {
        .mem_size = ctx_bytes,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    ctx = ggml_init(params);
    if (ctx == NULL) {
        fprintf(stderr, "ggml_init failed for %zu bytes\n", ctx_bytes);
        tq_runner_result_free(&result);
        return 1;
    }

    if (tq_ggml_materialize_f32(ctx, &result.tensor, &loaded, err, sizeof(err)) != 0) {
        fprintf(stderr, "loader materialize failed: %s\n", err);
        ggml_free(ctx);
        tq_runner_result_free(&result);
        return 1;
    }
    if (tq_ggml_write_loader_audit_json(&result, &loaded, audit_path, err, sizeof(err)) != 0) {
        fprintf(stderr, "loader audit failed: %s\n", err);
        ggml_free(ctx);
        tq_runner_result_free(&result);
        return 1;
    }

    printf("ggml loader surface complete\n");
    printf("  backend: %s\n", result.backend_name);
    printf("  tensor: %s\n", ggml_get_name(loaded.tensor));
    printf("  ggml_type: %s\n", ggml_type_name(loaded.tensor->type));
    printf(
        "  ggml_ne: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
        loaded.tensor->ne[0],
        loaded.tensor->ne[1],
        loaded.tensor->ne[2],
        loaded.tensor->ne[3]
    );
    printf(
        "  ggml_nb: [%zu, %zu, %zu, %zu]\n",
        loaded.tensor->nb[0],
        loaded.tensor->nb[1],
        loaded.tensor->nb[2],
        loaded.tensor->nb[3]
    );
    printf("  nbytes: %zu\n", loaded.nbytes);
    printf("  byte_identity: %s\n", loaded.byte_identity_ok ? "true" : "false");
    printf("  audit: %s\n", audit_path);

    ggml_free(ctx);
    tq_runner_result_free(&result);
    return 0;
}