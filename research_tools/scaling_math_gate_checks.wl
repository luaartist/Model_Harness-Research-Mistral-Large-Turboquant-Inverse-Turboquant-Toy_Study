(* Math checks for q4 shardlet catalog scaling. *)

ClearAll[
  valuesPerByte,
  payloadBytes,
  sourcePiBytes,
  runtimePiBytes,
  runtimeQjlMatrixBytes,
  centroidBytes,
  baselineBf16Bytes,
  sourceBytes,
  runtimeUploadBytes,
  payloadOnlyRatio,
  sourceRatio,
  runtimeUploadRatio,
  firstRowsForSourceRatio
];

valuesPerByte[mseBits_Integer] := Which[
  mseBits == 1, 8,
  mseBits == 2, 4,
  True, 2
];

payloadBytes[rows_Integer, dim_Integer, bits_Integer] := Module[
  {mseBits = bits - 1},
  rows * (
    Ceiling[dim / valuesPerByte[mseBits]] +
    Ceiling[dim / 8] +
    4
  )
];

sourcePiBytes[dim_Integer] := 2 * dim * dim;
runtimePiBytes[dim_Integer] := 4 * dim * dim;
runtimeQjlMatrixBytes[dim_Integer] := 4 * dim * dim;
centroidBytes[bits_Integer] := 4 * 2^(bits - 1);
baselineBf16Bytes[rows_Integer, dim_Integer] := 2 * rows * dim;

sourceBytes[rows_Integer, dim_Integer, bits_Integer] :=
  payloadBytes[rows, dim, bits] + sourcePiBytes[dim];

runtimeUploadBytes[rows_Integer, dim_Integer, bits_Integer] :=
  payloadBytes[rows, dim, bits] +
  runtimePiBytes[dim] +
  runtimeQjlMatrixBytes[dim] +
  centroidBytes[bits];

payloadOnlyRatio[rows_Integer, dim_Integer, bits_Integer] := N[
  baselineBf16Bytes[rows, dim] / payloadBytes[rows, dim, bits],
  16
];

sourceRatio[rows_Integer, dim_Integer, bits_Integer] := N[
  baselineBf16Bytes[rows, dim] / sourceBytes[rows, dim, bits],
  16
];

runtimeUploadRatio[rows_Integer, dim_Integer, bits_Integer] := N[
  baselineBf16Bytes[rows, dim] / runtimeUploadBytes[rows, dim, bits],
  16
];

firstRowsForSourceRatio[target_Real, dim_Integer, bits_Integer] := Module[
  {rows},
  rows = SelectFirst[
    Range[1, 100000],
    sourceRatio[#, dim, bits] >= target &,
    Missing["not_found"]
  ];
  rows
];

dim = 128;
bits = 4;
rowCounts = {256, 512, 1024, 2048, 4096, 8192};
targets = {1.0, 1.5, 2.0, 2.5, 2.9, 3.0};

report = <|
  "schema_version" -> 1,
  "purpose" -> "q4 shardlet catalog byte-accounting checks",
  "parameters" -> <|
    "vector_dim" -> dim,
    "key_bits" -> bits,
    "mse_bits" -> bits - 1,
    "values_per_mse_index_byte" -> valuesPerByte[bits - 1]
  |>,
  "per_vector" -> <|
    "payload_bytes" -> payloadBytes[1, dim, bits],
    "baseline_bf16_bytes" -> baselineBf16Bytes[1, dim],
    "payload_only_compression_ratio" -> payloadOnlyRatio[1, dim, bits]
  |>,
  "fixed_overheads" -> <|
    "source_pi_bytes_float16" -> sourcePiBytes[dim],
    "runtime_pi_bytes_float32" -> runtimePiBytes[dim],
    "runtime_qjl_matrix_bytes_float32" -> runtimeQjlMatrixBytes[dim],
    "centroid_bytes_float32" -> centroidBytes[bits]
  |>,
  "row_count_table" -> Table[
    <|
      "rows" -> rows,
      "baseline_bf16_bytes" -> baselineBf16Bytes[rows, dim],
      "payload_bytes" -> payloadBytes[rows, dim, bits],
      "source_bytes_with_pi" -> sourceBytes[rows, dim, bits],
      "runtime_upload_bytes" -> runtimeUploadBytes[rows, dim, bits],
      "payload_only_ratio" -> payloadOnlyRatio[rows, dim, bits],
      "source_ratio_with_pi" -> sourceRatio[rows, dim, bits],
      "runtime_upload_ratio" -> runtimeUploadRatio[rows, dim, bits]
    |>,
    {rows, rowCounts}
  ],
  "source_ratio_row_thresholds" -> Association@Table[
    ToString[target] -> firstRowsForSourceRatio[target, dim, bits],
    {target, targets}
  ],
  "notes" -> {
    "Payload-only q4 ratio is asymptotic and ignores pi overhead.",
    "Source ratio includes stored float16 pi per shardlet.",
    "Runtime upload ratio includes float32 pi, generated QJL matrix, and centroids."
  }
|>;

resultPath = FileNameJoin[{DirectoryName[$InputFileName], "results",
    "scaling_math_gate_checks.json"}];
CreateDirectory[DirectoryName[resultPath], CreateIntermediateDirectories -> True];
Export[resultPath, report, "JSON"];
Print["wrote: ", resultPath];
