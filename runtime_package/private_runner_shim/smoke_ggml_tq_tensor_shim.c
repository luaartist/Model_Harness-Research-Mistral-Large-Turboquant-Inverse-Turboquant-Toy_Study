#include "ggml_tq_tensor_shim.h"

#include <stdio.h>

int main(int argc, char **argv) {
    char err[512];
    tq_runner_backend_kind backend;
    tq_runner_result result;
    const char *input_path = argc > 1 ? argv[1] : "../../results/ggml_opencl_sidecar_decode.f32";
    const char *audit_path = argc > 2 ? argv[2] : "../../results/private_runner_api_audit.json";
    const char *backend_arg = argc > 3 ? argv[3] : "decoded_f32_file";
    const char *manifest_path = argc > 4 ? argv[4] : "../runtime_package_manifest.json";
    const char *backend_url = argc > 5 ? argv[5] : "http://127.0.0.1:8504";
    const char *output_path = argc > 6 ? argv[6] : "";
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
        .audit_path = audit_path,
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

    printf("private runner API complete\n");
    printf("  backend: %s\n", result.backend_name);
    printf("  tensor: %s\n", result.tensor.desc.name);
    printf(
        "  ggml_ne: [%zu, %zu, %zu, %zu]\n",
        result.tensor.desc.ne[0],
        result.tensor.desc.ne[1],
        result.tensor.desc.ne[2],
        result.tensor.desc.ne[3]
    );
    printf(
        "  ggml_nb: [%zu, %zu, %zu, %zu]\n",
        result.tensor.desc.nb[0],
        result.tensor.desc.nb[1],
        result.tensor.desc.nb[2],
        result.tensor.desc.nb[3]
    );
    printf("  nbytes: %zu\n", result.tensor.nbytes);
    printf("  l2_norm: %.17g\n", result.tensor.l2_norm);
    printf("  audit: %s\n", audit_path);

    if (output_path[0] != '\0') {
        FILE *output = fopen(output_path, "wb");
        if (output == NULL) {
            fprintf(stderr, "failed to open output: %s\n", output_path);
            tq_runner_result_free(&result);
            return 1;
        }
        if (fwrite(result.tensor.data, 1, result.tensor.nbytes, output) != result.tensor.nbytes) {
            fclose(output);
            fprintf(stderr, "failed to write output: %s\n", output_path);
            tq_runner_result_free(&result);
            return 1;
        }
        fclose(output);
        printf("  output: %s\n", output_path);
    }

    tq_runner_result_free(&result);
    return 0;
}