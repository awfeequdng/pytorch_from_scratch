# WARNING: the contents of this file must BOTH be valid Starlark (for Buck and

# Bazel) as well as valid Python (for our cmake build).  This means that
# load() directives are not allowed (as they are not recognized by Python).
# If you want to fix this, figure out how run this file from cmake with a proper
# Starlark interpreter as part of the default OSS build process.  If you need
# some nontrivial Starlark features, make a separate bzl file (remember that

# bzl files are not exported via ShipIt by default, so you may also need to
# update PyTorch's ShipIt config)

# This is duplicated in caffe2/CMakeLists.txt for now and not yet used in buck
GENERATED_LAZY_TS_CPP = [
]

# NVFuser runtime library
libtorch_nvfuser_runtime_sources = [

]

libtorch_nvfuser_generated_headers = ["{}.h".format(name.split("/")[-1].split(".")[0]) for name in libtorch_nvfuser_runtime_sources]

def libtorch_generated_sources(gencode_pattern):
    return [gencode_pattern.format(name) for name in [

    ]]

# copied from https://github.com/pytorch/pytorch/blob/f99a693cd9ff7a9b5fdc71357dac66b8192786d3/aten/src/ATen/core/CMakeLists.txt
jit_core_headers = [
]

jit_core_sources = [
]

# copied from https://github.com/pytorch/pytorch/blob/0bde610c14b92d351b968a0228df29e92442b1cc/torch/CMakeLists.txt
# There are some common files used in both internal lite-interpreter and full-jit. Making a separate
# list for the shared files.

core_sources_common = [
]

torch_unpickler_common = [
]

libtorch_sources_common = sorted(core_sources_common + torch_unpickler_common)

# The profilers are not needed in the lite interpreter build.
libtorch_profiler_sources = [
]

libtorch_edge_profiler_sources = libtorch_profiler_sources + [
]

core_trainer_sources = [
]

core_sources_full_mobile_no_backend_interface = [
]

core_sources_full_mobile = core_sources_full_mobile_no_backend_interface + [
]

core_sources_full = core_sources_full_mobile + [
]

lazy_tensor_core_sources = [
]

# We can't build all of the ts backend under certain build configurations, e.g. mobile,
# since it depends on things like autograd, meta functions, which may be disabled
lazy_tensor_ts_sources = [
]

lazy_tensor_core_python_sources = [
]

libtorch_core_sources = sorted(
    core_sources_common +
    torch_unpickler_common +
    core_sources_full +
    core_trainer_sources +
    libtorch_profiler_sources +
    lazy_tensor_core_sources,
)

# These files are the only ones that are supported on Windows.
libtorch_distributed_base_sources = [

]

# These files are only supported on Linux (and others) but not on Windows.
libtorch_distributed_extra_sources = [

]

libtorch_distributed_sources = libtorch_distributed_base_sources + libtorch_distributed_extra_sources

jit_sources_full = [

]

libtorch_core_jit_sources = sorted(jit_sources_full)

torch_mobile_tracer_sources = [

]

torch_mobile_core = [

]

libtorch_lite_eager_symbolication = [

]

# TODO: core_trainer_sources is not necessary for libtorch lite
libtorch_lite_cmake_sources = sorted(
    core_trainer_sources +
    core_sources_common +
    torch_unpickler_common +
    torch_mobile_core,
)

libtorch_cmake_sources = libtorch_core_sources + libtorch_core_jit_sources

libtorch_extra_sources = libtorch_core_jit_sources + [

]

def libtorch_sources(gencode_pattern = ":generate-code[{}]"):
    enable_flatbuffer = bool(native.read_config("fbcode", "caffe2_enable_flatbuffer", None))
    flatbuffer_serializer_sources = [
    ]
    if enable_flatbuffer:
        return (
            libtorch_generated_sources(gencode_pattern) + libtorch_core_sources + libtorch_distributed_sources + libtorch_extra_sources +
            flatbuffer_serializer_sources
        )
    else:
        return libtorch_generated_sources(gencode_pattern) + libtorch_core_sources + libtorch_distributed_sources + libtorch_extra_sources

libtorch_cuda_core_sources = [

]

# These files are the only ones that are supported on Windows.
libtorch_cuda_distributed_base_sources = [
]

# These files are only supported on Linux (and others) but not on Windows.
libtorch_cuda_distributed_extra_sources = [
]

libtorch_cuda_distributed_sources = libtorch_cuda_distributed_base_sources + libtorch_cuda_distributed_extra_sources

libtorch_cuda_sources = libtorch_cuda_core_sources + libtorch_cuda_distributed_sources + [
]

torch_cpp_srcs = [
]

libtorch_python_cuda_core_sources = [

]

libtorch_python_cuda_sources = libtorch_python_cuda_core_sources + [
]

libtorch_python_core_sources = [
    "torch/csrc/Module.cpp",
    "torch/csrc/utils.cpp",
] + lazy_tensor_core_python_sources

libtorch_python_distributed_core_sources = [
]

libtorch_python_distributed_sources = libtorch_python_distributed_core_sources + [

]

def glob_libtorch_python_sources(gencode_pattern = ":generate-code[{}]"):
    _libtorch_python_sources = [gencode_pattern.format(name) for name in [

    ]]

    _libtorch_python_sources.extend(libtorch_python_core_sources)
    _libtorch_python_sources.extend(libtorch_python_distributed_sources)

    return _libtorch_python_sources

aten_cpu_source_non_codegen_list = [

]

aten_cpu_source_codegen_list = [
]

aten_ufunc_headers = [
]

# When building lite interpreter in OSS, "aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp" will go through
# codegen process. The codegen version of this file, like Activation.cpp.DEFAULT.cpp, will be included
# in ${cpu_kernel_cpp} in aten/src/ATen/CMakeLists.txt. As a result, in aten/src/ATen/CMakeLists.txt,
# only aten_cpu_source_non_codegen_list need to be added to ${all_cpu_cpp}.
aten_cpu_source_list = sorted(aten_cpu_source_non_codegen_list + aten_cpu_source_codegen_list)

# Same as ${aten_cpu_source_codegen_list}, this list will go through aten codegen, and be included in
# ${cpu_kernel_cpp} in aten/src/ATen/CMakeLists.txt.
aten_native_source_codegen_list = [

]

# This aten native source file list will not go through aten codegen process
aten_native_source_non_codegen_list = [

]

# 1. Files in ATen/native with a few exceptions
# TODO: move the exceptions to proper locations
# 2. The whole aten native source list includes the list with and without aten codegen process.
aten_native_source_list = sorted(aten_native_source_non_codegen_list + aten_native_source_codegen_list)

# These are cpp files which need to go in the torch_cuda_cu library
# .cu files can be found via glob
aten_cuda_cu_source_list = [

]

# Files using thrust::sort_by_key need to be linked last
aten_cuda_with_sort_by_key_source_list = [
]

aten_cuda_cu_with_sort_by_key_source_list = [
]
