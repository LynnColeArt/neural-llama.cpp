#include "ggml-coreml.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-metal.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define GGML_COREML_NAME "COREML"

struct ggml_backend_coreml_backend {
    ggml_backend_t metal_backend;
};

struct ggml_backend_coreml_device {
    ggml_backend_dev_t metal_dev;
    std::string name;
    std::string description;
};

struct ggml_backend_coreml_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

using ggml_coreml_backend_context_t = ggml_backend_coreml_backend *;
using ggml_coreml_device_context_t = ggml_backend_coreml_device *;
using ggml_coreml_reg_context_t = ggml_backend_coreml_reg_context *;

extern const ggml_backend_i ggml_backend_coreml_i;

static ggml_guid_t ggml_backend_coreml_guid(void) {
    static ggml_guid guid = {
        0x5b, 0x61, 0x72, 0x8a, 0x32, 0xe3, 0x48, 0x13,
        0xad, 0x60, 0xc9, 0xb8, 0x6a, 0x70, 0x5b, 0xa8,
    };
    return &guid;
}

static ggml_backend_t ggml_backend_coreml_from_metal_backend(ggml_backend_t backend) {
    auto * backend_ctx = (ggml_coreml_backend_context_t)backend->context;
    return backend_ctx->metal_backend;
}

static bool ggml_coreml_output_contains_npu(const std::string & output) {
    if (output.find("aneDevicePropertyNumANECores") != std::string::npos) {
        const size_t pos = output.find("aneDevicePropertyNumANECores");
        const size_t eq = output.find('=', pos);
        if (eq != std::string::npos) {
            size_t i = eq + 1;
            while (i < output.size() && !std::isdigit(static_cast<unsigned char>(output[i]))) {
                ++i;
            }
            if (i < output.size()) {
                int value = 0;
                while (i < output.size() && std::isdigit(static_cast<unsigned char>(output[i]))) {
                    value = (value * 10) + (output[i] - '0');
                    ++i;
                }
                if (value > 0) {
                    return true;
                }
            }
        }
    }

    if (output.find("aneDevicePropertyANEVersion") != std::string::npos) {
        return true;
    }

    if (output.find("ANEHWDevice") != std::string::npos) {
        return true;
    }

    return false;
}

static bool ggml_coreml_probe_ioreg_npu() {
    static const std::array<const char *, 3> classes = {
        "H11ANEIn",
        "ANEHWDevice",
        "ANEClientDevice",
    };

    for (const auto & class_name : classes) {
        const std::string cmd = std::string("ioreg -r -c ") + class_name + " -d1 2>/dev/null";
        FILE * pipe = popen(cmd.c_str(), "r");
        if (pipe == nullptr) {
            continue;
        }

        std::string output;
        char buffer[1024];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }

        const int rc = pclose(pipe);
        if (rc != 0) {
            continue;
        }

        if (ggml_coreml_output_contains_npu(output)) {
            return true;
        }
    }

    return false;
}

static bool ggml_coreml_is_npu_available(void) {
    static bool initialized = false;
    static bool available = false;
    if (initialized) {
        return available;
    }

    initialized = true;

#if defined(__APPLE__)
    available = ggml_coreml_probe_ioreg_npu();
#else
    available = false;
#endif

    return available;
}

static const char * ggml_backend_coreml_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_COREML_NAME;
}

static void ggml_backend_coreml_free(ggml_backend_t backend) {
    auto * backend_ctx = (ggml_coreml_backend_context_t)backend->context;
    if (backend_ctx != NULL && backend_ctx->metal_backend != NULL) {
        ggml_backend_free(backend_ctx->metal_backend);
        backend_ctx->metal_backend = NULL;
    }
    delete backend_ctx;
    free(backend);
}

static void ggml_backend_coreml_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_tensor_set_async(ggml_backend_coreml_from_metal_backend(backend), tensor, data, offset, size);
}

static void ggml_backend_coreml_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_tensor_get_async(ggml_backend_coreml_from_metal_backend(backend), tensor, data, offset, size);
}

static bool ggml_backend_coreml_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_tensor_copy_async(
        ggml_backend_coreml_from_metal_backend(backend_src),
        ggml_backend_coreml_from_metal_backend(backend_dst),
        const_cast<ggml_tensor *>(src),
        dst
    );

    return true;
}

static void ggml_backend_coreml_synchronize(ggml_backend_t backend) {
    ggml_backend_synchronize(ggml_backend_coreml_from_metal_backend(backend));
}

static ggml_backend_graph_plan_t ggml_backend_coreml_graph_plan_create(ggml_backend_t backend, const ggml_cgraph * cgraph) {
    return ggml_backend_graph_plan_create(
        ggml_backend_coreml_from_metal_backend(backend),
        const_cast<ggml_cgraph *>(cgraph)
    );
}

static void ggml_backend_coreml_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ggml_backend_graph_plan_free(ggml_backend_coreml_from_metal_backend(backend), plan);
}

static void ggml_backend_coreml_graph_plan_update(ggml_backend_t backend, ggml_backend_graph_plan_t plan, const ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    GGML_UNUSED(plan);
    GGML_UNUSED(cgraph);
}

static enum ggml_status ggml_backend_coreml_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    return ggml_backend_graph_plan_compute(ggml_backend_coreml_from_metal_backend(backend), plan);
}

static enum ggml_status ggml_backend_coreml_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    return ggml_backend_graph_compute(ggml_backend_coreml_from_metal_backend(backend), cgraph);
}

static void ggml_backend_coreml_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    GGML_UNUSED(cgraph);
}

static void ggml_backend_coreml_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_event_record(event, ggml_backend_coreml_from_metal_backend(backend));
}

static void ggml_backend_coreml_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_event_wait(ggml_backend_coreml_from_metal_backend(backend), event);
}

const ggml_backend_i ggml_backend_coreml_i = {
    /* .get_name           = */ ggml_backend_coreml_name,
    /* .free               = */ ggml_backend_coreml_free,
    /* .set_tensor_async   = */ ggml_backend_coreml_set_tensor_async,
    /* .get_tensor_async   = */ ggml_backend_coreml_get_tensor_async,
    /* .cpy_tensor_async   = */ ggml_backend_coreml_cpy_tensor_async,
    /* .synchronize        = */ ggml_backend_coreml_synchronize,
    /* .graph_plan_create  = */ ggml_backend_coreml_graph_plan_create,
    /* .graph_plan_free    = */ ggml_backend_coreml_graph_plan_free,
    /* .graph_plan_update  = */ ggml_backend_coreml_graph_plan_update,
    /* .graph_plan_compute = */ ggml_backend_coreml_graph_plan_compute,
    /* .graph_compute      = */ ggml_backend_coreml_graph_compute,
    /* .event_record       = */ ggml_backend_coreml_event_record,
    /* .event_wait         = */ ggml_backend_coreml_event_wait,
    /* .graph_optimize     = */ ggml_backend_coreml_graph_optimize,
};

static const char * ggml_backend_coreml_device_get_name(ggml_backend_dev_t dev) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    return dev_ctx->name.c_str();
}

static const char * ggml_backend_coreml_device_get_description(ggml_backend_dev_t dev) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    return dev_ctx->description.c_str();
}

static void ggml_backend_coreml_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    ggml_backend_dev_memory(dev_ctx->metal_dev, free, total);
}

static enum ggml_backend_dev_type ggml_backend_coreml_device_get_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    GGML_UNUSED(dev_ctx);
    // CoreML delegates to Apple GPUs/ANE and should participate in default GPU offload
    // selection alongside Metal when present.
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_coreml_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;

    ggml_backend_dev_props delegated {};
    ggml_backend_dev_get_props(dev_ctx->metal_dev, &delegated);

    props->name = dev_ctx->name.c_str();
    props->description = dev_ctx->description.c_str();
    props->memory_free = delegated.memory_free;
    props->memory_total = delegated.memory_total;
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = delegated.device_id;
    props->caps = delegated.caps;
}

static ggml_backend_t ggml_backend_coreml_init_from_dev(ggml_backend_dev_t dev, const char * params) {
    auto * coreml_ctx = (ggml_coreml_device_context_t)dev->context;
    ggml_backend_t metal_backend = ggml_backend_dev_init(coreml_ctx->metal_dev, params);
    if (metal_backend == NULL) {
        return NULL;
    }

    auto * backend_ctx = new ggml_backend_coreml_backend { .metal_backend = metal_backend };
    auto * backend = (ggml_backend_t) malloc(sizeof(ggml_backend));
    *backend = {
        /* .guid      = */ ggml_backend_coreml_guid(),
        /* .iface     = */ ggml_backend_coreml_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    return backend;
}

static ggml_backend_t ggml_backend_coreml_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_coreml_init_from_dev(dev, params);
}

static ggml_backend_buffer_type_t ggml_backend_coreml_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    return ggml_backend_dev_buffer_type(dev_ctx->metal_dev);
}

static ggml_backend_buffer_type_t ggml_backend_coreml_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    return ggml_backend_dev_host_buffer_type(dev_ctx->metal_dev);
}

static ggml_backend_buffer_t ggml_backend_coreml_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    return ggml_backend_dev_buffer_from_host_ptr(dev_ctx->metal_dev, ptr, size, max_tensor_size);
}

static bool ggml_backend_coreml_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    return ggml_backend_dev_supports_op(dev_ctx->metal_dev, op);
}

static bool ggml_backend_coreml_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    return ggml_backend_dev_supports_buft(dev_ctx->metal_dev, buft);
}

static bool ggml_backend_coreml_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    return ggml_backend_dev_offload_op(dev_ctx->metal_dev, op);
}

static ggml_backend_event_t ggml_backend_coreml_device_event_new(ggml_backend_dev_t dev) {
    auto * coreml_dev = (ggml_coreml_device_context_t)dev->context;
    ggml_backend_event_t event = ggml_backend_event_new(coreml_dev->metal_dev);
    if (event == NULL) {
        return NULL;
    }
    return event;
}

static void ggml_backend_coreml_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    if (event != NULL) {
        ggml_backend_event_free(event);
    }
}

static void ggml_backend_coreml_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    if (event != NULL) {
        ggml_backend_event_synchronize(event);
    }
}

static const ggml_backend_device_i ggml_backend_coreml_device_i = {
    /* .get_name             = */ ggml_backend_coreml_device_get_name,
    /* .get_description      = */ ggml_backend_coreml_device_get_description,
    /* .get_memory           = */ ggml_backend_coreml_device_get_memory,
    /* .get_type             = */ ggml_backend_coreml_device_get_type,
    /* .get_props            = */ ggml_backend_coreml_device_get_props,
    /* .init_backend         = */ ggml_backend_coreml_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_coreml_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_coreml_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ggml_backend_coreml_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_coreml_device_supports_op,
    /* .supports_buft        = */ ggml_backend_coreml_device_supports_buft,
    /* .offload_op           = */ ggml_backend_coreml_device_offload_op,
    /* .event_new            = */ ggml_backend_coreml_device_event_new,
    /* .event_free           = */ ggml_backend_coreml_device_event_free,
    /* .event_synchronize    = */ ggml_backend_coreml_device_event_synchronize,
};

static const char * ggml_backend_coreml_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_COREML_NAME;
}

static size_t ggml_backend_coreml_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * reg_ctx = (ggml_coreml_reg_context_t)reg->context;
    return reg_ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_coreml_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * reg_ctx = (ggml_coreml_reg_context_t)reg->context;
    GGML_ASSERT(index < reg_ctx->devices.size());
    return reg_ctx->devices[index];
}

static void * ggml_backend_coreml_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    ggml_backend_reg_t metal_reg = ggml_backend_metal_reg();
    if (metal_reg == NULL) {
        return NULL;
    }
    return ggml_backend_reg_get_proc_address(metal_reg, name);
}

#ifdef GGML_USE_METAL
static const ggml_backend_reg_i ggml_backend_coreml_reg_i = {
    /* .get_name         = */ ggml_backend_coreml_reg_get_name,
    /* .get_device_count = */ ggml_backend_coreml_reg_get_device_count,
    /* .get_device       = */ ggml_backend_coreml_reg_get_device,
    /* .get_proc_address = */ ggml_backend_coreml_get_proc_address,
};
#endif

static ggml_backend_dev_t ggml_backend_coreml_device_new(ggml_backend_reg_t reg, int index, ggml_backend_dev_t metal_dev) {
    auto * coreml_dev = new ggml_backend_coreml_device {
        /* .metal_dev   = */ metal_dev,
        /* .name        = */ std::string(GGML_COREML_NAME) + std::to_string(index),
        /* .description = */ std::string("Apple Silicon NPU (CoreML backend)"),
    };

    return new ggml_backend_device {
        /* .iface   = */ ggml_backend_coreml_device_i,
        /* .reg     = */ reg,
        /* .context = */ coreml_dev,
    };
}

static void ggml_backend_coreml_device_free(ggml_backend_dev_t dev) {
    auto * dev_ctx = (ggml_coreml_device_context_t)dev->context;
    delete dev_ctx;
    delete dev;
}

struct ggml_coreml_device_deleter {
    void operator()(ggml_backend_dev_t dev) const {
        ggml_backend_coreml_device_free(dev);
    }
};

using ggml_coreml_device_ptr = std::unique_ptr<ggml_backend_device, ggml_coreml_device_deleter>;

struct ggml_coreml_reg_deleter {
    void operator()(ggml_coreml_reg_context_t ctx) const {
        delete ctx;
    }
};

using ggml_coreml_reg_ptr = std::unique_ptr<ggml_backend_coreml_reg_context, ggml_coreml_reg_deleter>;

ggml_backend_reg_t ggml_backend_coreml_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    if (!ggml_coreml_is_npu_available()) {
        return NULL;
    }

    {
        static std::mutex coreml_lock;
        std::lock_guard<std::mutex> lock_guard(coreml_lock);
        static ggml_coreml_reg_ptr reg_ctx;
        static std::vector<ggml_coreml_device_ptr> device_storage;

        if (!initialized) {
#ifdef GGML_USE_METAL
            ggml_backend_reg_t metal_reg = ggml_backend_metal_reg();
            if (metal_reg == NULL) {
                return NULL;
            }

            auto * coreml_reg = new ggml_backend_coreml_reg_context;
            reg_ctx.reset(coreml_reg);

            const size_t metal_device_count = ggml_backend_reg_dev_count(metal_reg);
            coreml_reg->devices.reserve(metal_device_count);
            for (size_t i = 0; i < metal_device_count; ++i) {
                ggml_backend_dev_t metal_dev = ggml_backend_reg_dev_get(metal_reg, i);
                auto * dev = ggml_backend_coreml_device_new(&reg, static_cast<int>(i), metal_dev);
                device_storage.emplace_back(dev);
                coreml_reg->devices.push_back(dev);
            }

            if (coreml_reg->devices.empty()) {
                return NULL;
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_coreml_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
#else
            return NULL;
#endif

            initialized = true;
        }
    }

    return &reg;
}

ggml_backend_t ggml_backend_coreml_init(void) {
    ggml_backend_reg_t reg = ggml_backend_coreml_reg();
    if (reg == NULL) {
        return NULL;
    }

    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (dev == NULL) {
        return NULL;
    }
    return ggml_backend_dev_init(dev, NULL);
}

bool ggml_backend_is_coreml(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_coreml_guid());
}

GGML_BACKEND_DL_IMPL(ggml_backend_coreml_reg)
