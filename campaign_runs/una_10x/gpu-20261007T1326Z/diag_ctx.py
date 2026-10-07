import ctypes
lib = ctypes.CDLL("libcuda.so.1")
print("cuInit rc=", lib.cuInit(0))
dev = ctypes.c_int(0)
print("cuDeviceGet rc=", lib.cuDeviceGet(ctypes.byref(dev), 0), "dev=", dev.value)
ctx = ctypes.c_void_p()
rc = lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
print("PrimaryCtxRetain rc=", rc, "ctx=", hex(ctx.value or 0))
print("CtxSetCurrent rc=", lib.cuCtxSetCurrent(ctx))
cur = ctypes.c_void_p()
print("CtxGetCurrent rc=", lib.cuCtxGetCurrent(ctypes.byref(cur)), "cur=", hex(cur.value or 0))
gd = ctypes.c_int(0)
print("CtxGetDevice rc=", lib.cuCtxGetDevice(ctypes.byref(gd)), "device=", gd.value)
ptr = ctypes.c_void_p()
lib.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
rc = lib.cuMemAlloc(ctypes.byref(ptr), ctypes.c_size_t(1024))
print("cuMemAlloc rc=", rc, "ptr=", hex(ptr.value or 0))
if rc != 0:
    print("cuGetLastErrorName? trying GetErrorString:")
    lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    s = ctypes.c_char_p()
    lib.cuGetErrorString(rc, ctypes.byref(s))
    print("  meaning:", s.value)
    # retry after re-setcurrent
    print("re-setcur rc=", lib.cuCtxSetCurrent(ctx))
    rc2 = lib.cuMemAlloc(ctypes.byref(ptr), ctypes.c_size_t(1024))
    print("cuMemAlloc retry rc=", rc2)
