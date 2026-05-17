CC ?= cc
LLAMA_CPP_DIR ?= ../../third_party/llama.cpp
GGML_BUILD_DIR ?= $(LLAMA_CPP_DIR)/build/bin
CURL_CFLAGS := $(shell pkg-config --cflags libcurl 2>/dev/null)
CURL_LIBS := $(shell pkg-config --libs libcurl 2>/dev/null || printf '%s' '-lcurl')
GGML_CFLAGS := -I$(LLAMA_CPP_DIR)/ggml/include
GGML_LIBS := -L$(GGML_BUILD_DIR) -Wl,-rpath,$(GGML_BUILD_DIR) -lggml -lggml-base -lggml-cpu
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -pedantic $(CURL_CFLAGS) $(GGML_CFLAGS)
LDFLAGS ?= -lm $(CURL_LIBS)

TARGET := smoke_ggml_tq_tensor_shim
LOADER_TARGET := smoke_ggml_tq_loader_adapter
OBJS := smoke_ggml_tq_tensor_shim.o ggml_tq_tensor_shim.o ggml_tq_http_bridge.o
LOADER_OBJS := smoke_ggml_tq_loader_adapter.o ggml_tq_loader_adapter.o ggml_tq_tensor_shim.o ggml_tq_http_bridge.o

.PHONY: all clean

all: $(TARGET) $(LOADER_TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(LOADER_TARGET): $(LOADER_OBJS)
	$(CC) $(CFLAGS) -o $@ $(LOADER_OBJS) $(LDFLAGS) $(GGML_LIBS)

%.o: %.c ggml_tq_tensor_shim.h
	$(CC) $(CFLAGS) -c $< -o $@

ggml_tq_loader_adapter.o smoke_ggml_tq_loader_adapter.o: ggml_tq_loader_adapter.h

clean:
	rm -f $(TARGET) $(LOADER_TARGET) $(OBJS) $(LOADER_OBJS)