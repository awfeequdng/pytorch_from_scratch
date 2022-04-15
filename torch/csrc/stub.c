#include <Python.h>

#ifdef _cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C(void);

extern PyObject* initModule(void);

PyMODINIT_FUNC PyInit__C(void) {
    return initModule();
}