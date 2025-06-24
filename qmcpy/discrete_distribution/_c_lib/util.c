#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "export_ctypes.h"

EXPORT int get_unsigned_long_size() { return sizeof(unsigned long); }
EXPORT int get_unsigned_long_long_size() { return sizeof(unsigned long long); }