
__kernel void turboquant_inverse_q4(
    __global const uchar * mse_packed,
    __global const uchar * qjl_packed,
    __global const float * norms,
    __global const float * residual_norms,
    __global const float * centroids,
    __global const float * pi,
    __global const float * s_matrix,
    __global float * output,
    const int rows,
    const int mse_packed_cols,
    const int qjl_packed_cols,
    const int vector_dim,
    const float qjl_scale
) {
    int gid = get_global_id(0);
    int total = rows * vector_dim;
    if (gid >= total) {
        return;
    }

    int row = gid / vector_dim;
    int col = gid - row * vector_dim;

    float mse_acc = 0.0f;
    float qjl_acc = 0.0f;
    for (int j = 0; j < vector_dim; ++j) {
        uchar packed_byte = mse_packed[row * mse_packed_cols + (j >> 1)];
        uchar code = (j & 1) == 0 ?
            (packed_byte & 15) : ((packed_byte >> 4) & 15);
        float y = centroids[(int) code];
        mse_acc += y * pi[j * vector_dim + col];

        uchar sign_byte = qjl_packed[row * qjl_packed_cols + (j >> 3)];
        float sign = ((sign_byte >> (j & 7)) & 1) ? 1.0f : -1.0f;
        qjl_acc += sign * s_matrix[j * vector_dim + col];
    }

    float mse_part = mse_acc * norms[row];
    float qjl_part = qjl_acc * qjl_scale * residual_norms[row];
    output[gid] = mse_part + qjl_part;
}
