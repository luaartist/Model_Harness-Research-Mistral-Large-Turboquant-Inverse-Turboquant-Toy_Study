#ifndef GGML_TQ_HTTP_BRIDGE_H
#define GGML_TQ_HTTP_BRIDGE_H

#include "ggml_tq_tensor_shim.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int tq_http_bridge_decode(
    const tq_runner_request *request,
    tq_runner_result *result,
    char *err,
    size_t err_cap
);

#ifdef __cplusplus
}
#endif

#endif