import ctypes
lib = ctypes.CDLL("libcuda.so.1")
print("cuInit rc=", lib.cuInit(0))
dev = ctypes.c_int(0)
print("cuDeviceGet rc=", lib.cuDeviceGet(ctypes.byref(dev), 0))

# state of the primary context
flags = ctypes.c_uint(0); active = ctypes.c_int(0)
lib.cuDevicePrimaryCtxGetState.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
rc = lib.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(active))
print("PrimaryCtxGetState rc=", rc, "flags=", flags.value, "active=", active.value)

# explicit context
ctx = ctypes.c_void_p()
lib.cuCtxCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
rc = lib.cuCtxCreate(ctypes.byref(ctx), ctypes.c_uint(0), dev)
print("cuCtxCreate rc=", rc, "ctx=", hex(ctx.value or 0))
cur = ctypes.c_void_p()
print("CtxGetCurrent rc=", lib.cuCtxGetCurrent(ctypes.byref(cur)), "cur=", hex(cur.value or 0))
ptr = ctypes.c_void_p()
lib.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
rc = lib.cuMemAlloc(ctypes.byref(ptr), ctypes.c_size_t(1024))
print("cuMemAlloc(explicit ctx) rc=", rc, "ptr=", hex(ptr.value or 0))
if ptr.value:
    lib.cuMemFree.argtypes = [ctypes.c_void_p]
    print("cuMemFree rc=", lib.cuMemFree(ptr))
lib.cuCtxDestroy.argtypes = [ctypes.c_void_p]
print("cuCtxDestroy rc=", lib.cuCtxDestroy(ctx))
