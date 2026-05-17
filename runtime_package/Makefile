PYTHON ?= /usr/bin/python3
RESULTS_DIR := ../results
PRIVATE_RUNNER_DIR := private_runner_shim
DECODE_F32 := $(RESULTS_DIR)/ggml_opencl_sidecar_decode.f32
RUNNER_AUDIT := $(RESULTS_DIR)/private_runner_api_audit.json
LOADER_AUDIT := $(RESULTS_DIR)/ggml_loader_surface_audit.json

.PHONY: verify validate sidecar descriptor c-smoke loader-smoke framework release-check release-lock journal-status journal-context journal-run-all gguf-write gguf-merge clean

verify: validate sidecar descriptor c-smoke loader-smoke framework release-check

validate:
	$(PYTHON) validate_runtime_package.py

sidecar:
	$(PYTHON) ggml_opencl_sidecar.py

descriptor:
	$(PYTHON) ggml_tensor_buffer_shim.py

c-smoke:
	$(MAKE) -C $(PRIVATE_RUNNER_DIR) clean all
	cd $(PRIVATE_RUNNER_DIR) && \
	  ./smoke_ggml_tq_tensor_shim \
	    ../../results/ggml_opencl_sidecar_decode.f32 \
	    ../../results/private_runner_api_audit.json \
	    decoded_f32_file

loader-smoke:
	$(MAKE) -C $(PRIVATE_RUNNER_DIR) all
	cd $(PRIVATE_RUNNER_DIR) && \
	  ./smoke_ggml_tq_loader_adapter \
	    ../../results/ggml_opencl_sidecar_decode.f32 \
	    ../../results/ggml_loader_surface_audit.json \
	    decoded_f32_file

framework:
	$(PYTHON) runner_framework.py --stable-audit

release-check:
	$(PYTHON) release_lock.py --check

release-lock:
	$(PYTHON) release_lock.py --write

journal-status:
	$(PYTHON) workflow_journal.py status

journal-context:
	$(PYTHON) workflow_journal.py context

journal-run-all:
	$(PYTHON) workflow_journal.py run-all

clean:
	$(MAKE) -C $(PRIVATE_RUNNER_DIR) clean

TENSOR_NAME ?= blk.0.attn_q.weight

gguf-write:
	$(PYTHON) gguf_shardlet_writer.py --tensor-name "$(TENSOR_NAME)"

gguf-merge:
	$(PYTHON) gguf_multi_tensor_merger.py

