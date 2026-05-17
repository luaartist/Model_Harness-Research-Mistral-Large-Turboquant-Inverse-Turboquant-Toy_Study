#include "ggml_tq_http_bridge.h"

#include <curl/curl.h>

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct tq_http_manifest {
    char artifact_path[4096];
    char prefix[256];
    char kernel_path[4096];
    char kernel_function[128];
    char centroids_path[4096];
    char qjl_matrix_path[4096];
    size_t rows;
    size_t vector_dim;
    size_t key_bits;
    size_t mse_bits;
    double qjl_scale;
} tq_http_manifest;

typedef struct tq_http_buffer {
    unsigned char *data;
    size_t size;
} tq_http_buffer;

typedef struct tq_st_info {
    char dtype[16];
    size_t shape[4];
    size_t ndim;
    uint64_t offset_begin;
    uint64_t offset_end;
} tq_st_info;

static int bridge_error(char *err, size_t err_cap, const char *msg) {
    if (err != NULL && err_cap > 0) {
        snprintf(err, err_cap, "%s", msg != NULL ? msg : "unknown error");
    }
    return -1;
}

static int bridge_errorf(char *err, size_t err_cap, const char *fmt, const char *value) {
    if (err != NULL && err_cap > 0) {
        snprintf(err, err_cap, fmt, value != NULL ? value : "");
    }
    return -1;
}

static void tq_http_buffer_free(tq_http_buffer *buffer) {
    if (buffer != NULL) {
        free(buffer->data);
        buffer->data = NULL;
        buffer->size = 0;
    }
}

static int read_entire_file(const char *path, unsigned char **out, size_t *out_size, char *err, size_t err_cap) {
    FILE *handle = NULL;
    long size_long = 0;
    size_t size = 0;
    unsigned char *data = NULL;

    if (path == NULL || path[0] == '\0') {
        return bridge_error(err, err_cap, "file path is empty");
    }
    handle = fopen(path, "rb");
    if (handle == NULL) {
        if (err != NULL && err_cap > 0) {
            snprintf(err, err_cap, "failed to open %s: %s", path, strerror(errno));
        }
        return -1;
    }
    if (fseek(handle, 0, SEEK_END) != 0) {
        fclose(handle);
        return bridge_errorf(err, err_cap, "failed to seek %s", path);
    }
    size_long = ftell(handle);
    if (size_long < 0) {
        fclose(handle);
        return bridge_errorf(err, err_cap, "failed to size %s", path);
    }
    if (fseek(handle, 0, SEEK_SET) != 0) {
        fclose(handle);
        return bridge_errorf(err, err_cap, "failed to rewind %s", path);
    }
    size = (size_t)size_long;
    data = (unsigned char *)malloc(size > 0 ? size : 1);
    if (data == NULL) {
        fclose(handle);
        return bridge_error(err, err_cap, "out of memory while reading file");
    }
    if (size > 0 && fread(data, 1, size, handle) != size) {
        free(data);
        fclose(handle);
        return bridge_errorf(err, err_cap, "short read from %s", path);
    }
    fclose(handle);
    *out = data;
    *out_size = size;
    return 0;
}

static int read_text_file(const char *path, char **out, size_t *out_size, char *err, size_t err_cap) {
    unsigned char *data = NULL;
    size_t size = 0;
    char *text = NULL;
    if (read_entire_file(path, &data, &size, err, err_cap) != 0) {
        return -1;
    }
    text = (char *)malloc(size + 1);
    if (text == NULL) {
        free(data);
        return bridge_error(err, err_cap, "out of memory while reading text file");
    }
    memcpy(text, data, size);
    text[size] = '\0';
    free(data);
    *out = text;
    if (out_size != NULL) {
        *out_size = size;
    }
    return 0;
}

static char *dup_text(const char *value) {
    size_t len = value != NULL ? strlen(value) : 0;
    char *copy = (char *)malloc(len + 1);
    if (copy == NULL) {
        return NULL;
    }
    if (len > 0) {
        memcpy(copy, value, len);
    }
    copy[len] = '\0';
    return copy;
}

static const char *skip_ws(const char *cursor) {
    while (cursor != NULL && *cursor != '\0' && isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    return cursor;
}

static const char *range_find_quoted_key(const char *start, const char *end, const char *key) {
    size_t key_len = strlen(key);
    const char *cursor = start;
    while (cursor != NULL && cursor < end) {
        const char *found = (const char *)memchr(cursor, '"', (size_t)(end - cursor));
        if (found == NULL || found + key_len + 1 >= end) {
            return NULL;
        }
        if (memcmp(found + 1, key, key_len) == 0 && found[1 + key_len] == '"') {
            return found;
        }
        cursor = found + 1;
    }
    return NULL;
}

static const char *find_matching_brace(const char *open) {
    int depth = 0;
    int in_string = 0;
    int escaped = 0;
    const char *cursor = open;
    while (*cursor != '\0') {
        const char c = *cursor;
        if (in_string) {
            if (escaped) {
                escaped = 0;
            } else if (c == '\\') {
                escaped = 1;
            } else if (c == '"') {
                in_string = 0;
            }
        } else if (c == '"') {
            in_string = 1;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                return cursor + 1;
            }
        }
        ++cursor;
    }
    return NULL;
}

static int json_find_object_range(const char *json, const char *key, const char **start, const char **end) {
    const char *key_pos = range_find_quoted_key(json, json + strlen(json), key);
    const char *colon = NULL;
    const char *open = NULL;
    const char *close = NULL;
    if (key_pos == NULL) {
        return -1;
    }
    colon = strchr(key_pos, ':');
    if (colon == NULL) {
        return -1;
    }
    open = strchr(colon, '{');
    if (open == NULL) {
        return -1;
    }
    close = find_matching_brace(open);
    if (close == NULL) {
        return -1;
    }
    *start = open;
    *end = close;
    return 0;
}

static int json_string_in_range(
    const char *start,
    const char *end,
    const char *field,
    char *out,
    size_t out_cap
) {
    const char *key_pos = range_find_quoted_key(start, end, field);
    const char *colon = NULL;
    const char *cursor = NULL;
    size_t used = 0;
    if (key_pos == NULL) {
        return -1;
    }
    colon = memchr(key_pos, ':', (size_t)(end - key_pos));
    if (colon == NULL) {
        return -1;
    }
    cursor = skip_ws(colon + 1);
    if (cursor == NULL || cursor >= end || *cursor != '"') {
        return -1;
    }
    ++cursor;
    while (cursor < end && *cursor != '"') {
        if (*cursor == '\\' && cursor + 1 < end) {
            ++cursor;
        }
        if (used + 1 >= out_cap) {
            return -1;
        }
        out[used++] = *cursor++;
    }
    if (cursor >= end || *cursor != '"') {
        return -1;
    }
    out[used] = '\0';
    return 0;
}

static int json_string_in_object(
    const char *json,
    const char *object_key,
    const char *field,
    char *out,
    size_t out_cap
) {
    const char *start = NULL;
    const char *end = NULL;
    if (json_find_object_range(json, object_key, &start, &end) != 0) {
        return -1;
    }
    return json_string_in_range(start, end, field, out, out_cap);
}

static int json_string_global(const char *json, const char *field, char *out, size_t out_cap) {
    return json_string_in_range(json, json + strlen(json), field, out, out_cap);
}

static int json_number_global(const char *json, const char *field, double *out) {
    const char *end = json + strlen(json);
    const char *key_pos = range_find_quoted_key(json, end, field);
    const char *colon = NULL;
    char *number_end = NULL;
    if (key_pos == NULL) {
        return -1;
    }
    colon = memchr(key_pos, ':', (size_t)(end - key_pos));
    if (colon == NULL) {
        return -1;
    }
    *out = strtod(skip_ws(colon + 1), &number_end);
    return number_end != skip_ws(colon + 1) ? 0 : -1;
}

static int json_size_array2_in_object(
    const char *json,
    const char *object_key,
    const char *field,
    size_t *first,
    size_t *second
) {
    const char *start = NULL;
    const char *end = NULL;
    const char *key_pos = NULL;
    const char *colon = NULL;
    const char *bracket = NULL;
    char *parse_end = NULL;
    unsigned long long a = 0;
    unsigned long long b = 0;

    if (json_find_object_range(json, object_key, &start, &end) != 0) {
        return -1;
    }
    key_pos = range_find_quoted_key(start, end, field);
    if (key_pos == NULL) {
        return -1;
    }
    colon = memchr(key_pos, ':', (size_t)(end - key_pos));
    if (colon == NULL) {
        return -1;
    }
    bracket = memchr(colon, '[', (size_t)(end - colon));
    if (bracket == NULL) {
        return -1;
    }
    a = strtoull(skip_ws(bracket + 1), &parse_end, 10);
    if (parse_end == skip_ws(bracket + 1)) {
        return -1;
    }
    parse_end = (char *)skip_ws(parse_end);
    if (*parse_end != ',') {
        return -1;
    }
    b = strtoull(skip_ws(parse_end + 1), &parse_end, 10);
    if (parse_end == skip_ws(parse_end + 1)) {
        return -1;
    }
    *first = (size_t)a;
    *second = (size_t)b;
    return 0;
}

static int json_offsets_in_object(const char *json, const char *tensor_name, uint64_t *begin, uint64_t *finish) {
    size_t first = 0;
    size_t second = 0;
    if (json_size_array2_in_object(json, tensor_name, "data_offsets", &first, &second) != 0) {
        return -1;
    }
    *begin = (uint64_t)first;
    *finish = (uint64_t)second;
    return 0;
}

static int tq_manifest_load(const char *path, tq_http_manifest *manifest, char *err, size_t err_cap) {
    char *json = NULL;
    double number = 0.0;
    size_t rows = 0;
    size_t mse_cols = 0;
    (void)mse_cols;

    memset(manifest, 0, sizeof(*manifest));
    if (read_text_file(path, &json, NULL, err, err_cap) != 0) {
        return -1;
    }
    if (json_string_in_object(json, "artifact", "path", manifest->artifact_path, sizeof(manifest->artifact_path)) != 0 ||
        json_string_global(json, "prefix", manifest->prefix, sizeof(manifest->prefix)) != 0 ||
        json_string_in_object(json, "kernel", "path", manifest->kernel_path, sizeof(manifest->kernel_path)) != 0 ||
        json_string_in_object(json, "kernel", "function", manifest->kernel_function, sizeof(manifest->kernel_function)) != 0 ||
        json_string_in_object(json, "centroids", "path", manifest->centroids_path, sizeof(manifest->centroids_path)) != 0 ||
        json_string_in_object(json, "qjl_matrix", "path", manifest->qjl_matrix_path, sizeof(manifest->qjl_matrix_path)) != 0) {
        free(json);
        return bridge_error(err, err_cap, "manifest is missing required path/function fields");
    }
    if (json_number_global(json, "vector_dim", &number) != 0) {
        free(json);
        return bridge_error(err, err_cap, "manifest missing vector_dim");
    }
    manifest->vector_dim = (size_t)number;
    if (json_number_global(json, "key_bits", &number) != 0) {
        free(json);
        return bridge_error(err, err_cap, "manifest missing key_bits");
    }
    manifest->key_bits = (size_t)number;
    if (json_number_global(json, "mse_bits", &number) != 0) {
        free(json);
        return bridge_error(err, err_cap, "manifest missing mse_bits");
    }
    manifest->mse_bits = (size_t)number;
    if (json_number_global(json, "qjl_scale", &manifest->qjl_scale) != 0) {
        free(json);
        return bridge_error(err, err_cap, "manifest missing qjl_scale");
    }
    if (json_size_array2_in_object(json, "mse_indices", "shape", &rows, &mse_cols) != 0) {
        free(json);
        return bridge_error(err, err_cap, "manifest missing mse_indices shape");
    }
    manifest->rows = rows;
    free(json);
    return 0;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t f = 0;
    float out = 0.0f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    memcpy(&out, &f, sizeof(out));
    return out;
}

static uint64_t read_u64_le(const unsigned char *data) {
    uint64_t value = 0;
    for (int shift = 0; shift < 8; ++shift) {
        value |= ((uint64_t)data[shift]) << (8 * shift);
    }
    return value;
}

static int safetensors_tensor_info(
    const char *header,
    const char *tensor_name,
    tq_st_info *info,
    char *err,
    size_t err_cap
) {
    size_t dim0 = 0;
    size_t dim1 = 1;
    memset(info, 0, sizeof(*info));
    if (json_string_in_object(header, tensor_name, "dtype", info->dtype, sizeof(info->dtype)) != 0 ||
        json_offsets_in_object(header, tensor_name, &info->offset_begin, &info->offset_end) != 0) {
        return bridge_errorf(err, err_cap, "missing safetensors metadata for %s", tensor_name);
    }
    if (json_size_array2_in_object(header, tensor_name, "shape", &dim0, &dim1) == 0) {
        info->ndim = 2;
        info->shape[0] = dim0;
        info->shape[1] = dim1;
    } else {
        const char *start = NULL;
        const char *end = NULL;
        const char *key_pos = NULL;
        const char *colon = NULL;
        const char *bracket = NULL;
        char *parse_end = NULL;
        if (json_find_object_range(header, tensor_name, &start, &end) != 0) {
            return bridge_errorf(err, err_cap, "missing safetensors object for %s", tensor_name);
        }
        key_pos = range_find_quoted_key(start, end, "shape");
        if (key_pos == NULL || (colon = memchr(key_pos, ':', (size_t)(end - key_pos))) == NULL ||
            (bracket = memchr(colon, '[', (size_t)(end - colon))) == NULL) {
            return bridge_errorf(err, err_cap, "missing shape for %s", tensor_name);
        }
        dim0 = (size_t)strtoull(skip_ws(bracket + 1), &parse_end, 10);
        if (parse_end == skip_ws(bracket + 1)) {
            return bridge_errorf(err, err_cap, "invalid shape for %s", tensor_name);
        }
        info->ndim = 1;
        info->shape[0] = dim0;
    }
    return 0;
}

static int safetensors_read_tensor(
    const char *artifact_path,
    const char *tensor_name,
    tq_st_info *info,
    unsigned char **raw,
    size_t *raw_size,
    char *err,
    size_t err_cap
) {
    unsigned char *file = NULL;
    size_t file_size = 0;
    uint64_t header_len = 0;
    uint64_t data_start = 0;
    char *header = NULL;
    uint64_t begin = 0;
    uint64_t finish = 0;
    size_t nbytes = 0;
    unsigned char *payload = NULL;

    if (read_entire_file(artifact_path, &file, &file_size, err, err_cap) != 0) {
        return -1;
    }
    if (file_size < 8) {
        free(file);
        return bridge_error(err, err_cap, "safetensors file too small");
    }
    header_len = read_u64_le(file);
    data_start = 8 + header_len;
    if (header_len > file_size || data_start > file_size) {
        free(file);
        return bridge_error(err, err_cap, "invalid safetensors header length");
    }
    header = (char *)malloc((size_t)header_len + 1);
    if (header == NULL) {
        free(file);
        return bridge_error(err, err_cap, "out of memory for safetensors header");
    }
    memcpy(header, file + 8, (size_t)header_len);
    header[header_len] = '\0';
    if (safetensors_tensor_info(header, tensor_name, info, err, err_cap) != 0) {
        free(header);
        free(file);
        return -1;
    }
    begin = data_start + info->offset_begin;
    finish = data_start + info->offset_end;
    if (finish < begin || finish > file_size) {
        free(header);
        free(file);
        return bridge_errorf(err, err_cap, "invalid data offsets for %s", tensor_name);
    }
    nbytes = (size_t)(finish - begin);
    payload = (unsigned char *)malloc(nbytes > 0 ? nbytes : 1);
    if (payload == NULL) {
        free(header);
        free(file);
        return bridge_error(err, err_cap, "out of memory for tensor payload");
    }
    memcpy(payload, file + begin, nbytes);
    free(header);
    free(file);
    *raw = payload;
    *raw_size = nbytes;
    return 0;
}

static int load_safetensors_u8(
    const char *artifact_path,
    const char *tensor_name,
    size_t expected_bytes,
    unsigned char **out,
    char *err,
    size_t err_cap
) {
    tq_st_info info;
    unsigned char *raw = NULL;
    size_t raw_size = 0;
    if (safetensors_read_tensor(artifact_path, tensor_name, &info, &raw, &raw_size, err, err_cap) != 0) {
        return -1;
    }
    if (strcmp(info.dtype, "U8") != 0 || raw_size != expected_bytes) {
        free(raw);
        return bridge_errorf(err, err_cap, "unexpected U8 tensor layout for %s", tensor_name);
    }
    *out = raw;
    return 0;
}

static int load_safetensors_f32(
    const char *artifact_path,
    const char *tensor_name,
    size_t expected_count,
    float **out,
    char *err,
    size_t err_cap
) {
    tq_st_info info;
    unsigned char *raw = NULL;
    size_t raw_size = 0;
    float *values = NULL;
    if (safetensors_read_tensor(artifact_path, tensor_name, &info, &raw, &raw_size, err, err_cap) != 0) {
        return -1;
    }
    values = (float *)malloc(expected_count * sizeof(float));
    if (values == NULL) {
        free(raw);
        return bridge_error(err, err_cap, "out of memory for f32 tensor");
    }
    if (strcmp(info.dtype, "F32") == 0) {
        if (raw_size != expected_count * sizeof(float)) {
            free(values);
            free(raw);
            return bridge_errorf(err, err_cap, "unexpected F32 tensor size for %s", tensor_name);
        }
        memcpy(values, raw, raw_size);
    } else if (strcmp(info.dtype, "F16") == 0) {
        if (raw_size != expected_count * sizeof(uint16_t)) {
            free(values);
            free(raw);
            return bridge_errorf(err, err_cap, "unexpected F16 tensor size for %s", tensor_name);
        }
        for (size_t index = 0; index < expected_count; ++index) {
            const uint16_t half = (uint16_t)raw[index * 2] | ((uint16_t)raw[index * 2 + 1] << 8);
            values[index] = f16_to_f32(half);
        }
    } else {
        free(values);
        free(raw);
        return bridge_errorf(err, err_cap, "unsupported float dtype for %s", tensor_name);
    }
    free(raw);
    *out = values;
    return 0;
}

static int load_raw_f32_file(
    const char *path,
    size_t expected_count,
    float **out,
    char *err,
    size_t err_cap
) {
    unsigned char *raw = NULL;
    size_t raw_size = 0;
    float *values = NULL;
    if (read_entire_file(path, &raw, &raw_size, err, err_cap) != 0) {
        return -1;
    }
    if (raw_size != expected_count * sizeof(float)) {
        free(raw);
        return bridge_errorf(err, err_cap, "unexpected raw f32 size for %s", path);
    }
    values = (float *)malloc(raw_size);
    if (values == NULL) {
        free(raw);
        return bridge_error(err, err_cap, "out of memory for raw f32 data");
    }
    memcpy(values, raw, raw_size);
    free(raw);
    *out = values;
    return 0;
}

static size_t curl_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
    tq_http_buffer *buffer = (tq_http_buffer *)userdata;
    const size_t incoming = size * nmemb;
    unsigned char *next = (unsigned char *)realloc(buffer->data, buffer->size + incoming + 1);
    if (next == NULL) {
        return 0;
    }
    buffer->data = next;
    memcpy(buffer->data + buffer->size, ptr, incoming);
    buffer->size += incoming;
    buffer->data[buffer->size] = '\0';
    return incoming;
}

static int http_request(
    const char *url,
    const char *content_type,
    const unsigned char *body,
    size_t body_size,
    tq_http_buffer *response,
    char *err,
    size_t err_cap
) {
    CURL *curl = curl_easy_init();
    CURLcode code;
    long http_code = 0;
    struct curl_slist *headers = NULL;

    memset(response, 0, sizeof(*response));
    if (curl == NULL) {
        return bridge_error(err, err_cap, "curl_easy_init failed");
    }
    if (content_type != NULL) {
        char header[256];
        snprintf(header, sizeof(header), "Content-Type: %s", content_type);
        headers = curl_slist_append(headers, header);
    }
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    if (headers != NULL) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    if (body != NULL) {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)body_size);
    }

    code = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (headers != NULL) {
        curl_slist_free_all(headers);
    }
    curl_easy_cleanup(curl);
    if (code != CURLE_OK) {
        tq_http_buffer_free(response);
        return bridge_errorf(err, err_cap, "curl request failed: %s", curl_easy_strerror(code));
    }
    if (http_code < 200 || http_code >= 300) {
        tq_http_buffer_free(response);
        if (err != NULL && err_cap > 0) {
            snprintf(err, err_cap, "HTTP %ld from %s", http_code, url);
        }
        return -1;
    }
    return 0;
}

static int make_url(const char *base, const char *path, char *out, size_t out_cap) {
    const size_t base_len = strlen(base);
    const int has_slash = base_len > 0 && base[base_len - 1] == '/';
    snprintf(out, out_cap, "%s%s%s", base, has_slash || path[0] == '/' ? "" : "/", path);
    return 0;
}

static int json_get_i64(const unsigned char *json, const char *field, long long *out) {
    double number = 0.0;
    if (json_number_global((const char *)json, field, &number) != 0) {
        return -1;
    }
    *out = (long long)number;
    return 0;
}

static int bridge_status(const char *base_url, char *err, size_t err_cap) {
    char url[8192];
    tq_http_buffer response;
    make_url(base_url, "/status", url, sizeof(url));
    if (http_request(url, NULL, NULL, 0, &response, err, err_cap) != 0) {
        return -1;
    }
    tq_http_buffer_free(&response);
    return 0;
}

static int bridge_json_result(const tq_http_buffer *response, char *err, size_t err_cap) {
    long long result = 1;
    if (json_get_i64(response->data, "result", &result) != 0 || result != 0) {
        return bridge_error(err, err_cap, "bridge response result was nonzero or missing");
    }
    return 0;
}

static int bridge_malloc(const char *base_url, size_t size, long long *handle, char *err, size_t err_cap) {
    char url[8192];
    char payload[128];
    tq_http_buffer response;
    make_url(base_url, "/hip/malloc", url, sizeof(url));
    snprintf(payload, sizeof(payload), "{\"size\":%zu}", size);
    if (http_request(url, "application/json", (const unsigned char *)payload, strlen(payload), &response, err, err_cap) != 0) {
        return -1;
    }
    if (bridge_json_result(&response, err, err_cap) != 0 || json_get_i64(response.data, "handle", handle) != 0) {
        tq_http_buffer_free(&response);
        return bridge_error(err, err_cap, "hip/malloc response missing handle");
    }
    tq_http_buffer_free(&response);
    return 0;
}

static void bridge_free_quiet(const char *base_url, long long handle) {
    char url[8192];
    char payload[128];
    tq_http_buffer response;
    if (handle <= 0) {
        return;
    }
    make_url(base_url, "/hip/free", url, sizeof(url));
    snprintf(payload, sizeof(payload), "{\"handle\":%lld}", handle);
    if (http_request(url, "application/json", (const unsigned char *)payload, strlen(payload), &response, NULL, 0) == 0) {
        tq_http_buffer_free(&response);
    }
}

static int bridge_upload(
    const char *base_url,
    long long handle,
    const void *data,
    size_t nbytes,
    char *err,
    size_t err_cap
) {
    char url[8192];
    tq_http_buffer response;
    snprintf(url, sizeof(url), "%s/hip/memcpy_htod?handle=%lld", base_url, handle);
    if (http_request(url, "application/octet-stream", (const unsigned char *)data, nbytes, &response, err, err_cap) != 0) {
        return -1;
    }
    if (bridge_json_result(&response, err, err_cap) != 0) {
        tq_http_buffer_free(&response);
        return -1;
    }
    tq_http_buffer_free(&response);
    return 0;
}

static char *json_escape_alloc(const char *text) {
    const unsigned char *cursor = (const unsigned char *)(text != NULL ? text : "");
    const size_t len = strlen((const char *)cursor);
    char *out = (char *)malloc(len * 6 + 1);
    size_t used = 0;
    if (out == NULL) {
        return NULL;
    }
    while (*cursor != '\0') {
        const unsigned char c = *cursor++;
        if (c == '"' || c == '\\') {
            out[used++] = '\\';
            out[used++] = (char)c;
        } else if (c == '\n') {
            out[used++] = '\\';
            out[used++] = 'n';
        } else if (c == '\r') {
            out[used++] = '\\';
            out[used++] = 'r';
        } else if (c == '\t') {
            out[used++] = '\\';
            out[used++] = 't';
        } else if (c >= 0x20 && c < 0x7f) {
            out[used++] = (char)c;
        } else {
            snprintf(out + used, 7, "\\u%04x", (unsigned)c);
            used += 6;
        }
    }
    out[used] = '\0';
    return out;
}

static int bridge_module_load(
    const char *base_url,
    const char *kernel_source,
    long long *module,
    char *err,
    size_t err_cap
) {
    char url[8192];
    char *escaped = json_escape_alloc(kernel_source);
    char *payload = NULL;
    size_t payload_size = 0;
    tq_http_buffer response;
    if (escaped == NULL) {
        return bridge_error(err, err_cap, "out of memory escaping kernel source");
    }
    payload_size = strlen(escaped) + 32;
    payload = (char *)malloc(payload_size);
    if (payload == NULL) {
        free(escaped);
        return bridge_error(err, err_cap, "out of memory building module_load payload");
    }
    snprintf(payload, payload_size, "{\"source\":\"%s\"}", escaped);
    free(escaped);
    make_url(base_url, "/hip/module_load", url, sizeof(url));
    if (http_request(url, "application/json", (const unsigned char *)payload, strlen(payload), &response, err, err_cap) != 0) {
        free(payload);
        return -1;
    }
    free(payload);
    if (bridge_json_result(&response, err, err_cap) != 0 || json_get_i64(response.data, "module", module) != 0) {
        tq_http_buffer_free(&response);
        return bridge_error(err, err_cap, "module_load response missing module");
    }
    tq_http_buffer_free(&response);
    return 0;
}

static int bridge_module_get_function(
    const char *base_url,
    long long module,
    const char *function_name,
    long long *function,
    char *err,
    size_t err_cap
) {
    char url[8192];
    char payload[512];
    tq_http_buffer response;
    make_url(base_url, "/hip/module_get_function", url, sizeof(url));
    snprintf(payload, sizeof(payload), "{\"module\":%lld,\"name\":\"%s\"}", module, function_name);
    if (http_request(url, "application/json", (const unsigned char *)payload, strlen(payload), &response, err, err_cap) != 0) {
        return -1;
    }
    if (bridge_json_result(&response, err, err_cap) != 0 || json_get_i64(response.data, "function", function) != 0) {
        tq_http_buffer_free(&response);
        return bridge_error(err, err_cap, "module_get_function response missing function");
    }
    tq_http_buffer_free(&response);
    return 0;
}

static int bridge_launch_decode(
    const char *base_url,
    long long function,
    long long mse_handle,
    long long qjl_signs_handle,
    long long norms_handle,
    long long residual_handle,
    long long centroids_handle,
    long long pi_handle,
    long long qjl_matrix_handle,
    long long output_handle,
    size_t rows,
    size_t vector_dim,
    size_t mse_cols,
    size_t qjl_cols,
    double qjl_scale,
    char *err,
    size_t err_cap
) {
    char url[8192];
    char payload[4096];
    tq_http_buffer response;
    const size_t block = 128;
    const size_t grid = (rows * vector_dim + block - 1) / block;
    make_url(base_url, "/hip/launch_kernel", url, sizeof(url));
    snprintf(
        payload,
        sizeof(payload),
        "{\"function\":%lld,\"grid\":[%zu],\"block\":[%zu],\"args\":["
        "{\"kind\":\"buffer\",\"handle\":%lld},"
        "{\"kind\":\"buffer\",\"handle\":%lld},"
        "{\"kind\":\"buffer\",\"handle\":%lld},"
        "{\"kind\":\"buffer\",\"handle\":%lld},"
        "{\"kind\":\"buffer\",\"handle\":%lld},"
        "{\"kind\":\"buffer\",\"handle\":%lld},"
        "{\"kind\":\"buffer\",\"handle\":%lld},"
        "{\"kind\":\"buffer\",\"handle\":%lld},"
        "{\"kind\":\"i32\",\"value\":%zu},"
        "{\"kind\":\"i32\",\"value\":%zu},"
        "{\"kind\":\"i32\",\"value\":%zu},"
        "{\"kind\":\"i32\",\"value\":%zu},"
        "{\"kind\":\"f32\",\"value\":%.17g}]}",
        function,
        grid,
        block,
        mse_handle,
        qjl_signs_handle,
        norms_handle,
        residual_handle,
        centroids_handle,
        pi_handle,
        qjl_matrix_handle,
        output_handle,
        rows,
        mse_cols,
        qjl_cols,
        vector_dim,
        qjl_scale
    );
    if (http_request(url, "application/json", (const unsigned char *)payload, strlen(payload), &response, err, err_cap) != 0) {
        return -1;
    }
    if (bridge_json_result(&response, err, err_cap) != 0) {
        tq_http_buffer_free(&response);
        return -1;
    }
    tq_http_buffer_free(&response);
    return 0;
}

static int bridge_download(
    const char *base_url,
    long long handle,
    size_t size,
    tq_http_buffer *out,
    char *err,
    size_t err_cap
) {
    char url[8192];
    char payload[128];
    make_url(base_url, "/hip/memcpy_dtoh", url, sizeof(url));
    snprintf(payload, sizeof(payload), "{\"handle\":%lld,\"size\":%zu}", handle, size);
    if (http_request(url, "application/json", (const unsigned char *)payload, strlen(payload), out, err, err_cap) != 0) {
        return -1;
    }
    if (out->size != size) {
        tq_http_buffer_free(out);
        return bridge_error(err, err_cap, "downloaded byte count mismatch");
    }
    return 0;
}

static void compute_tensor_stats(tq_tensor_buffer *buffer, char *err, size_t err_cap) {
    float *values = (float *)buffer->data;
    const size_t count = buffer->nbytes / sizeof(float);
    double sum_sq = 0.0;
    buffer->sample_len = count < 8 ? count : 8;
    for (size_t index = 0; index < count; ++index) {
        const float value = values[index];
        if (!isfinite(value)) {
            bridge_error(err, err_cap, "http_bridge output contained non-finite values");
            return;
        }
        sum_sq += (double)value * (double)value;
        if (index < buffer->sample_len) {
            buffer->sample[index] = value;
        }
    }
    buffer->l2_norm = sqrt(sum_sq);
}

int tq_http_bridge_decode(
    const tq_runner_request *request,
    tq_runner_result *result,
    char *err,
    size_t err_cap
) {
    tq_http_manifest manifest;
    const char *base_url = NULL;
    char tensor_name[512];
    char *kernel_source = NULL;
    unsigned char *mse_indices = NULL;
    unsigned char *qjl_signs = NULL;
    float *norms = NULL;
    float *residual_norms = NULL;
    float *pi = NULL;
    float *centroids = NULL;
    float *qjl_matrix = NULL;
    tq_http_buffer output = {0};
    long long handles[8] = {0};
    long long module = 0;
    long long function = 0;
    size_t mse_cols = 0;
    size_t qjl_cols = 0;
    size_t output_nbytes = 0;
    char out_name[512];
    char out_path[512];

    if (request == NULL || result == NULL) {
        return bridge_error(err, err_cap, "http_bridge request/result is null");
    }
    if (request->package_manifest_path == NULL || request->package_manifest_path[0] == '\0') {
        return bridge_error(err, err_cap, "http_bridge requires package_manifest_path");
    }
    if (tq_manifest_load(request->package_manifest_path, &manifest, err, err_cap) != 0) {
        return -1;
    }
    base_url = request->backend_url != NULL && request->backend_url[0] != '\0'
        ? request->backend_url
        : "http://127.0.0.1:8504";

    curl_global_init(CURL_GLOBAL_DEFAULT);
    if (bridge_status(base_url, err, err_cap) != 0) {
        char status_error[512];
        snprintf(status_error, sizeof(status_error), "%s", err != NULL ? err : "");
        return bridge_errorf(err, err_cap, "http_bridge status failed: %s", status_error);
    }

    mse_cols = manifest.vector_dim / (8 / manifest.key_bits);
    qjl_cols = manifest.vector_dim / 8;
    output_nbytes = manifest.rows * manifest.vector_dim * sizeof(float);

    snprintf(tensor_name, sizeof(tensor_name), "%s.mse_indices", manifest.prefix);
    if (load_safetensors_u8(manifest.artifact_path, tensor_name, manifest.rows * mse_cols, &mse_indices, err, err_cap) != 0) {
        goto fail;
    }
    snprintf(tensor_name, sizeof(tensor_name), "%s.qjl_signs", manifest.prefix);
    if (load_safetensors_u8(manifest.artifact_path, tensor_name, manifest.rows * qjl_cols, &qjl_signs, err, err_cap) != 0) {
        goto fail;
    }
    snprintf(tensor_name, sizeof(tensor_name), "%s.norms", manifest.prefix);
    if (load_safetensors_f32(manifest.artifact_path, tensor_name, manifest.rows, &norms, err, err_cap) != 0) {
        goto fail;
    }
    snprintf(tensor_name, sizeof(tensor_name), "%s.residual_norms", manifest.prefix);
    if (load_safetensors_f32(manifest.artifact_path, tensor_name, manifest.rows, &residual_norms, err, err_cap) != 0) {
        goto fail;
    }
    snprintf(tensor_name, sizeof(tensor_name), "%s.pi", manifest.prefix);
    if (load_safetensors_f32(manifest.artifact_path, tensor_name, manifest.vector_dim * manifest.vector_dim, &pi, err, err_cap) != 0) {
        goto fail;
    }
    if (load_raw_f32_file(manifest.centroids_path, (size_t)1 << manifest.mse_bits, &centroids, err, err_cap) != 0 ||
        load_raw_f32_file(manifest.qjl_matrix_path, manifest.vector_dim * manifest.vector_dim, &qjl_matrix, err, err_cap) != 0 ||
        read_text_file(manifest.kernel_path, &kernel_source, NULL, err, err_cap) != 0) {
        goto fail;
    }

    if (bridge_module_load(base_url, kernel_source, &module, err, err_cap) != 0 ||
        bridge_module_get_function(base_url, module, manifest.kernel_function, &function, err, err_cap) != 0 ||
        bridge_malloc(base_url, manifest.rows * mse_cols, &handles[0], err, err_cap) != 0 ||
        bridge_upload(base_url, handles[0], mse_indices, manifest.rows * mse_cols, err, err_cap) != 0 ||
        bridge_malloc(base_url, manifest.rows * qjl_cols, &handles[1], err, err_cap) != 0 ||
        bridge_upload(base_url, handles[1], qjl_signs, manifest.rows * qjl_cols, err, err_cap) != 0 ||
        bridge_malloc(base_url, manifest.rows * sizeof(float), &handles[2], err, err_cap) != 0 ||
        bridge_upload(base_url, handles[2], norms, manifest.rows * sizeof(float), err, err_cap) != 0 ||
        bridge_malloc(base_url, manifest.rows * sizeof(float), &handles[3], err, err_cap) != 0 ||
        bridge_upload(base_url, handles[3], residual_norms, manifest.rows * sizeof(float), err, err_cap) != 0 ||
        bridge_malloc(base_url, ((size_t)1 << manifest.mse_bits) * sizeof(float), &handles[4], err, err_cap) != 0 ||
        bridge_upload(base_url, handles[4], centroids, ((size_t)1 << manifest.mse_bits) * sizeof(float), err, err_cap) != 0 ||
        bridge_malloc(base_url, manifest.vector_dim * manifest.vector_dim * sizeof(float), &handles[5], err, err_cap) != 0 ||
        bridge_upload(base_url, handles[5], pi, manifest.vector_dim * manifest.vector_dim * sizeof(float), err, err_cap) != 0 ||
        bridge_malloc(base_url, manifest.vector_dim * manifest.vector_dim * sizeof(float), &handles[6], err, err_cap) != 0 ||
        bridge_upload(base_url, handles[6], qjl_matrix, manifest.vector_dim * manifest.vector_dim * sizeof(float), err, err_cap) != 0 ||
        bridge_malloc(base_url, output_nbytes, &handles[7], err, err_cap) != 0 ||
        bridge_launch_decode(
            base_url,
            function,
            handles[0],
            handles[1],
            handles[2],
            handles[3],
            handles[4],
            handles[5],
            handles[6],
            handles[7],
            manifest.rows,
            manifest.vector_dim,
            mse_cols,
            qjl_cols,
            manifest.qjl_scale,
            err,
            err_cap
        ) != 0 ||
        bridge_download(base_url, handles[7], output_nbytes, &output, err, err_cap) != 0) {
        goto fail;
    }

    for (int index = 7; index >= 0; --index) {
        bridge_free_quiet(base_url, handles[index]);
        handles[index] = 0;
    }

    snprintf(out_name, sizeof(out_name), "%s.decoded_f32", manifest.prefix);
    snprintf(out_path, sizeof(out_path), "http_bridge://%s", manifest.prefix);
    result->tensor.desc.name = dup_text(out_name);
    result->tensor.desc.path = dup_text(out_path);
    result->tensor.desc.expected_sha256 = dup_text("");
    if (result->tensor.desc.name == NULL || result->tensor.desc.path == NULL || result->tensor.desc.expected_sha256 == NULL) {
        goto fail;
    }
    result->tensor.desc.rows = manifest.rows;
    result->tensor.desc.cols = manifest.vector_dim;
    result->tensor.desc.item_size = sizeof(float);
    result->tensor.desc.ne[0] = manifest.vector_dim;
    result->tensor.desc.ne[1] = manifest.rows;
    result->tensor.desc.ne[2] = 1;
    result->tensor.desc.ne[3] = 1;
    result->tensor.desc.nb[0] = sizeof(float);
    result->tensor.desc.nb[1] = manifest.vector_dim * sizeof(float);
    result->tensor.desc.nb[2] = output_nbytes;
    result->tensor.desc.nb[3] = output_nbytes;
    result->tensor.desc.nbytes = output_nbytes;
    result->tensor.data = output.data;
    result->tensor.nbytes = output.size;
    result->tensor.owns_desc_strings = 1;
    output.data = NULL;
    output.size = 0;
    compute_tensor_stats(&result->tensor, err, err_cap);
    result->used_http_bridge = 1;

    free(kernel_source);
    free(mse_indices);
    free(qjl_signs);
    free(norms);
    free(residual_norms);
    free(pi);
    free(centroids);
    free(qjl_matrix);
    tq_http_buffer_free(&output);
    return 0;

fail:
    for (int index = 7; index >= 0; --index) {
        bridge_free_quiet(base_url, handles[index]);
    }
    free(kernel_source);
    free(mse_indices);
    free(qjl_signs);
    free(norms);
    free(residual_norms);
    free(pi);
    free(centroids);
    free(qjl_matrix);
    tq_http_buffer_free(&output);
    return -1;
}