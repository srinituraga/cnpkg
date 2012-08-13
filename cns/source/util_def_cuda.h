/***********************************************************************************************************************
*
* Copyright (C) 2009 by Jim Mutch (www.jimmutch.com).
*
* This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later
* version.
*
* This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
* warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with this program.  If not, see
* <http://www.gnu.org/licenses/>.
*
***********************************************************************************************************************/

void _CudaExit(_Session *s, cudaError_t err, const char *format, ...) {

    char msg[_ERRMSG_LEN];

    va_list argList;
    va_start(argList, format);
    vsprintf(msg, format, argList);
    va_end(argList);

    sprintf(msg + strlen(msg), " (cuda error: %s)", cudaGetErrorString(err));

    s->Exit2("%s", msg);

}

/**********************************************************************************************************************/

void _ClaimDevice(_Session *s, int desiredDeviceNo, bool nice, _DeviceProps &props) {

    cudaError_t err;

    cudaThreadExit(); // Just to be safe.

    if (desiredDeviceNo >= 0) {
        if ((err = cudaSetDevice(desiredDeviceNo)) != cudaSuccess) {
            _CudaExit(s, err, "unable to select device %i", desiredDeviceNo);
        }
    }

    int flags = nice ? cudaDeviceBlockingSync : cudaDeviceScheduleSpin;
    if ((err = cudaSetDeviceFlags(flags)) != cudaSuccess) {
        _CudaExit(s, err, "unable to set device flags");
    }

    // This should be the first call that causes a new context to be created.
    void *ptr;
    if ((err = cudaMalloc(&ptr, 16)) != cudaSuccess) {
        _CudaExit(s, err, "device not available");
    }
    cudaFree(ptr);

    int deviceNo;
    if ((err = cudaGetDevice(&deviceNo)) != cudaSuccess) {
        _CudaExit(s, err, "unable to determine which device was activated");
    }

    if ((desiredDeviceNo >= 0) && (deviceNo != desiredDeviceNo)) {
        s->Exit2("wrong device activated (desired=%i, activated=%i)", desiredDeviceNo, deviceNo);
    }

    props.deviceNo = (unsigned int)deviceNo;

    // TODO: query device for this (the warp size).
    props.blockSizeAlign = 32;

    // Note: half the warp size.  Would like to query the device in case this is not appropriate, but don't know how.
    props.blockYSizeAlign = props.blockSizeAlign >> 1;

    // Note: would like to query the device for these properties, but don't know how.
    props.maxTexYSize = 1 << 16;
    props.maxTexXSize = 1 << 15;

}

/**********************************************************************************************************************/

void _ReleaseDevice() {

    cudaThreadExit();

}

/**********************************************************************************************************************/

void _ClearTex(_TexPtrA &ta, texture<float, 2> &tb) {

    ta.linear = false;
    ta.arr    = NULL;

}

/**********************************************************************************************************************/

void _AllocTexLinear(_Session *s, _TexPtrA &ta, texture<float, 2> &tb, const float *sBuf, unsigned int ySize,
    unsigned int xCount, const char *desc) {

    cudaError_t err;

    _ClearTex(ta, tb);

    if ((ySize == 0) || (xCount == 0)) return;

    ta.linear = true;

    tb.addressMode[0] = cudaAddressModeClamp;
    tb.addressMode[1] = cudaAddressModeClamp;
    tb.filterMode     = cudaFilterModePoint;
    tb.normalized     = false;
    cudaChannelFormatDesc channels = cudaCreateChannelDesc<float>();
    size_t offset;
    if ((err = cudaBindTexture2D(&offset, tb, sBuf, channels,
        ySize, xCount, ySize * sizeof(float))) != cudaSuccess) {
        ta.linear = false;
        _CudaExit(s, err, "unable to bind to %s buffer", desc);
    }
    if (offset != 0) {
        ta.linear = false;
        s->Exit2("nonzero offset returned while binding to %s buffer", desc);
    }

}

/**********************************************************************************************************************/

void _AllocTexArray(_Session *s, _TexPtrA &ta, texture<float, 2> &tb, const float *sBuf, unsigned int ySize,
    unsigned int xCount, const char *desc) {

    cudaError_t err;

    _ClearTex(ta, tb);

    if ((ySize == 0) || (xCount == 0)) return;

    cudaChannelFormatDesc channels = cudaCreateChannelDesc<float>();
    if ((err = cudaMallocArray(&ta.arr, &channels, ySize, xCount)) != cudaSuccess) {
        ta.arr = NULL;
        _CudaExit(s, err, "unable to allocate %s array", desc);
    }

    tb.addressMode[0] = cudaAddressModeClamp;
    tb.addressMode[1] = cudaAddressModeClamp;
    tb.filterMode     = cudaFilterModePoint;
    tb.normalized     = false;
    if ((err = cudaBindTextureToArray(tb, ta.arr)) != cudaSuccess) {
        cudaFreeArray(ta.arr);
        ta.arr = NULL;
        _CudaExit(s, err, "unable to bind to %s array", desc);
    }

    if (sBuf != NULL) {
        if ((err = cudaMemcpy2DToArray(
            ta.arr,
            0, 0,
            sBuf, ySize * sizeof(float),
            ySize * sizeof(float), xCount,
            cudaMemcpyHostToDevice)) != cudaSuccess) {
            _DeallocTex(ta, tb);
            _CudaExit(s, err, "unable to copy to %s array", desc);
        }
    }

}

/**********************************************************************************************************************/

void _DeallocTex(_TexPtrA &ta, texture<float, 2> &tb) {

    if (ta.linear) {

        cudaUnbindTexture(tb);

    } else if (ta.arr != NULL) {

        cudaUnbindTexture(tb);
        cudaFreeArray(ta.arr);

    }

    _ClearTex(ta, tb);

}

/**********************************************************************************************************************/

void _PublishTex(_Session *s, _TexPtrA &ta, texture<float, 2> &tb, const float *sBuf, unsigned int ySize,
    unsigned int xCount, const char *desc) {

    cudaError_t err;

    if (ta.linear) {
        s->Exit2("_PublishTex not implemented for linear memory textures");
    }

    if ((ySize == 0) || (xCount == 0)) return;

    if ((err = cudaMemcpy2DToArray(
        ta.arr,
        0, 0,
        sBuf, ySize * sizeof(float),
        ySize * sizeof(float), xCount,
        cudaMemcpyDeviceToDevice)) != cudaSuccess) {
        _CudaExit(s, err, "unable to copy to %s array", desc);
    }

}

/**********************************************************************************************************************/

void _CopyTex(_Session *s, float *dBuf, unsigned int dHeight, char dest, _TexPtrA &ta, texture<float, 2> &tb,
    unsigned int yOff, unsigned int xOff, unsigned int yCount, unsigned int xCount, const char *desc) {

    cudaError_t err;

    if (ta.linear) {
        s->Exit2("_CopyTex not implemented for linear memory textures");
    }

    if ((yCount == 0) || (xCount == 0)) return;

    enum cudaMemcpyKind kind = (dest == 'd') ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;

    if ((err = cudaMemcpy2DFromArray(
        dBuf, dHeight * sizeof(float),
        ta.arr, yOff * sizeof(float), xOff,
        yCount * sizeof(float), xCount,
        kind)) != cudaSuccess) {
        _CudaExit(s, err, "unable to copy %s array", desc);
    }

}

/**********************************************************************************************************************/

void _UpdateTex(_Session *s, _TexPtrA &ta, texture<float, 2> &tb, unsigned int yOff, unsigned int xOff,
    const float *sBuf, unsigned int sHeight, char src, unsigned int yCount, unsigned int xCount, const char *desc) {

    cudaError_t err;

    if (ta.linear) {
        s->Exit2("_UpdateTex not implemented for linear memory textures");
    }

    if ((yCount == 0) || (xCount == 0)) return;

    enum cudaMemcpyKind kind = (src == 'd') ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

    if ((err = cudaMemcpy2DToArray(
        ta.arr, yOff * sizeof(float), xOff,
        sBuf, sHeight * sizeof(float),
        yCount * sizeof(float), xCount,
        kind)) != cudaSuccess) {
        _CudaExit(s, err, "unable to copy to %s array", desc);
    }

}

/**********************************************************************************************************************/

template<class T> void _ClearBuf(T *&buf) {

    buf = NULL;

}

/**********************************************************************************************************************/

template<class T> void _AllocBuf(_Session *s, T *&buf, const T *sBuf, unsigned int numElements, const char *desc) {

    cudaError_t err;

    if (numElements == 0) {
        buf = NULL;
        return;
    }

    if ((err = cudaMalloc((void **)&buf, numElements * sizeof(T))) != cudaSuccess) {
        buf = NULL;
        _CudaExit(s, err, "unable to allocate %s buffer", desc);
    }

    if (sBuf != NULL) {
        if ((err = cudaMemcpy(
            buf, sBuf, numElements * sizeof(T), 
            cudaMemcpyHostToDevice)) != cudaSuccess) {
            _CudaExit(s, err, "unable to copy to %s buffer", desc);
        }
    }

}

/**********************************************************************************************************************/

template<class T> void _DeallocBuf(T *&buf) {

    if (buf != NULL) {
        cudaFree(buf);
        buf = NULL;
    }

}

/**********************************************************************************************************************/

template<class T> void _CopyBuf1D(_Session *s, T *dBuf, char dest, const T *buf, unsigned int count, const char *desc) {

    cudaError_t err;

    if (count == 0) return;

    enum cudaMemcpyKind kind = (dest == 'd') ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;

    if ((err = cudaMemcpy(
        dBuf, buf, count * sizeof(T), 
        kind)) != cudaSuccess) {
        _CudaExit(s, err, "unable to copy %s buffer", desc);
    }

}

/**********************************************************************************************************************/

template<class T> void _CopyBuf2D(_Session *s, T *dBuf, unsigned int dHeight, char dest, const T *buf,
    unsigned int height, unsigned int yCount, unsigned int xCount, const char *desc) {

    cudaError_t err;

    if ((yCount == 0) || (xCount == 0)) return;

    enum cudaMemcpyKind kind = (dest == 'd') ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;

    if ((err = cudaMemcpy2D(
        dBuf, dHeight * sizeof(T),
        buf, height * sizeof(T),
        yCount * sizeof(T), xCount,
        kind)) != cudaSuccess) {
        _CudaExit(s, err, "unable to copy %s buffer", desc);
    }

}

/**********************************************************************************************************************/

template<class T> void _CopyBuf3D(_Session *s, T *dBuf, unsigned int dHeight, unsigned int dWidth, char dest,
    const T *buf, unsigned int height, unsigned int width, unsigned int yOff, unsigned int xOff, unsigned int zOff,
    unsigned int yCount, unsigned int xCount, unsigned int zCount, const char *desc) {

    cudaError_t err;

    if (zCount == 1) {
        buf += (zOff * width + xOff) * height + yOff;
        _CopyBuf2D(s, dBuf, dHeight, dest, buf, height, yCount, xCount, desc);
        return;
    }

    if ((yCount == 0) || (xCount == 0) || (zCount == 0)) return;

    cudaMemcpy3DParms p = {0};

    p.srcPtr.ptr    = (T *)buf;
    p.srcPtr.pitch  = height * sizeof(T);
    p.srcPtr.xsize  = height * sizeof(T);
    p.srcPtr.ysize  = width;
    p.srcPos.x      = yOff;
    p.srcPos.y      = xOff;
    p.srcPos.z      = zOff;
    p.dstPtr.ptr    = dBuf;
    p.dstPtr.pitch  = dHeight * sizeof(T);
    p.dstPtr.xsize  = dHeight * sizeof(T);
    p.dstPtr.ysize  = dWidth;
    p.extent.width  = yCount * sizeof(T);
    p.extent.height = xCount;
    p.extent.depth  = zCount;
    p.kind          = (dest == 'd') ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;

    if ((err = cudaMemcpy3D(&p)) != cudaSuccess) {
        _CudaExit(s, err, "unable to copy %s buffer", desc);
    }

}

/**********************************************************************************************************************/

template<class T> void _UpdateBuf1D(_Session *s, T *buf, const T *sBuf, char src, unsigned int count,
    const char *desc) {

    cudaError_t err;

    if (count == 0) return;

    enum cudaMemcpyKind kind = (src == 'd') ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

    if ((err = cudaMemcpy(
        buf, sBuf, count * sizeof(T), 
        kind)) != cudaSuccess) {
        _CudaExit(s, err, "unable to update %s buffer", desc);
    }

}

/**********************************************************************************************************************/

template<class T> void _UpdateBuf2D(_Session *s, T *buf, unsigned int height, const T *sBuf, unsigned int sHeight,
    char src, unsigned int yCount, unsigned int xCount, const char *desc) {

    cudaError_t err;

    if ((yCount == 0) || (xCount == 0)) return;

    enum cudaMemcpyKind kind = (src == 'd') ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

    if ((err = cudaMemcpy2D(
        buf, height * sizeof(T),
        sBuf, sHeight * sizeof(T),
        yCount * sizeof(T), xCount,
        kind)) != cudaSuccess) {
        _CudaExit(s, err, "unable to update %s buffer", desc);
    }

}

/**********************************************************************************************************************/

template<class T> void _UpdateBuf3D(_Session *s, T *buf, unsigned int height, unsigned int width, unsigned int yOff,
    unsigned int xOff, unsigned int zOff, const T *sBuf, unsigned int sHeight, unsigned int sWidth, char src,
    unsigned int yCount, unsigned int xCount, unsigned int zCount, const char *desc) {

    cudaError_t err;

    if (zCount == 1) {
        buf += (zOff * width + xOff) * height + yOff;
        _UpdateBuf2D(s, buf, height, sBuf, sHeight, src, yCount, xCount, desc);
        return;
    }

    if ((yCount == 0) || (xCount == 0) || (zCount == 0)) return;

    cudaMemcpy3DParms p = {0};

    p.srcPtr.ptr    = (T *)sBuf;
    p.srcPtr.pitch  = sHeight * sizeof(T);
    p.srcPtr.xsize  = sHeight * sizeof(T);
    p.srcPtr.ysize  = sWidth;
    p.dstPtr.ptr    = buf;
    p.dstPtr.pitch  = height * sizeof(T);
    p.dstPtr.xsize  = height * sizeof(T);
    p.dstPtr.ysize  = width;
    p.dstPos.x      = yOff;
    p.dstPos.y      = xOff;
    p.dstPos.z      = zOff;
    p.extent.width  = yCount * sizeof(T);
    p.extent.height = xCount;
    p.extent.depth  = zCount;
    p.kind          = (src == 'd') ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

    if ((err = cudaMemcpy3D(&p)) != cudaSuccess) {
        _CudaExit(s, err, "unable to update %s buffer", desc);
    }

}

/**********************************************************************************************************************/

template<class T> void _AllocConst(_Session *s, const char *constName, T *cPtr, const T *sPtr,
    unsigned int numElements) {

    cudaError_t err;

    if (numElements == 0) return;

    if ((err = cudaMemcpyToSymbol(
        constName,
        sPtr,
        numElements * sizeof(T), 0,
        cudaMemcpyHostToDevice)) != cudaSuccess) {
        _CudaExit(s, err, "unable to set constant %s", constName);
    }

}

/**********************************************************************************************************************/

template<class T> void _CopyConst(_Session *s, T *dPtr, const char *constName, const T *cPtr, unsigned int off,
    unsigned int count) {

    cudaError_t err;

    if (count == 0) return;

    if ((err = cudaMemcpyFromSymbol(
        dPtr,
        constName,
        count * sizeof(T), off * sizeof(T),
        cudaMemcpyDeviceToHost)) != cudaSuccess) {
        _CudaExit(s, err, "unable to copy constant %s", constName);
    }

}

/**********************************************************************************************************************/

template<class T> void _UpdateConst(_Session *s, const char *constName, T *cPtr, unsigned int off, const T *sPtr,
    unsigned int count) {

    cudaError_t err;

    if (count == 0) return;

    if ((err = cudaMemcpyToSymbol(
        constName,
        sPtr,
        count * sizeof(T), off * sizeof(T),
        cudaMemcpyHostToDevice)) != cudaSuccess) {
        _CudaExit(s, err, "unable to update constant %s", constName);
    }

}
