#pragma once

#define EXPORT_API __attribute__((__visibility__("default")))
#define HIDDEN_API __attribute__((__visibility__("hidden")))

#define PXTORCH_API EXPORT_API

#define EXPORT EXPORT_API

#define PXML_CUDA_API EXPORT_API