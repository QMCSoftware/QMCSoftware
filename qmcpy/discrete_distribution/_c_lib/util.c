#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "export_ctypes.h"

#include <Python.h>

// in Windows, you must define an initialization function for your extension
// because setuptools will build a .pyd file, not a DLL
// https://stackoverflow.com/questions/34689210/error-exporting-symbol-when-building-python-c-extension-in-windows
PyMODINIT_FUNC PyInit__c_lib(void)
{
    printf("");
    return NULL;
}

EXPORT int get_unsigned_long_size() { return sizeof(unsigned long); }
EXPORT int get_unsigned_long_long_size() { return sizeof(unsigned long long); }