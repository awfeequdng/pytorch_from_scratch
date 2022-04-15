#include <Python.h>
#include <iostream>

#include <vector>
// #include <torch/csrc/utils.h>
void THPUtils_addPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods);

PyObject* module;

static std::vector<PyMethodDef> methods;

// extern "C" PyObject* initModule();

static PyObject *spam_system(PyObject* self, PyObject *args) {
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command)) {
        return NULL;
    }
    sts = system(command);
    return PyLong_FromLong(sts);
}

// Callback for python part. Used for additional initialization of python classes
static PyObject * THPModule_initExtension(PyObject *_unused, PyObject *shm_manager_path)
{
    std::cout << "THPModule_initExtension" << std::endl;
    Py_RETURN_NONE;
}

static PyMethodDef TorchMethods[] = {
    {"system", spam_system, METH_VARARGS, "Execute a shell command."},
    {"_initExtension",  THPModule_initExtension,   METH_O,       nullptr},
    {nullptr, nullptr, 0, nullptr}
};

extern "C" PyObject* initModule() {

#define ASSERT_TRUE(cmd) if (!(cmd)) return nullptr
    THPUtils_addPyMethodDefs(methods, TorchMethods);

    static struct PyModuleDef torchmodule = {
        PyModuleDef_HEAD_INIT,
        "torch._C",
        nullptr,
        -1,
        methods.data()
    };
    ASSERT_TRUE(module = PyModule_Create(&torchmodule));
    // ASSERT_TRUE(THPGenerator_init(module));
    return module;
// #undef ASSERT_TRUE
}