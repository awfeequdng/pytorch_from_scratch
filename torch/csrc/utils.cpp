#include <vector>
// #include <torch/csrc/utils.h>
#include <Python.h>

void THPUtils_addPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods);
void THPUtils_addPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods) {
    if (!vector.empty()) {
        // remove nullptr terminator
        vector.pop_back();
    }
    while (true) {
        vector.push_back(*methods);
        if (!methods->ml_name) {
            break;
        }
        methods++;
    }
}