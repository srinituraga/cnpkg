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

#ifdef _GPU

    typedef struct {
        bool       linear;
        cudaArray *arr;
    } _TexPtrA;

    #define _TEXTUREA _TexPtrA
    #define _TEXTUREB texture<float, 2>

    static void _CudaExit(_Session *s, cudaError_t err, const char *format, ...);

    #define INLINE  __device__
    #define INLINEH __host__ __device__

#else

    #include <math.h>

    typedef struct {unsigned short x, y, z, w;} ushort4;

    typedef struct {
        bool   linear;
        float *buf;
    } _TexPtrA;

    typedef struct {
        const float  *buf;
        unsigned int  h;
    } _TexPtrB;

    #define _TEXTUREA _TexPtrA
    #define _TEXTUREB _TexPtrB

    #define INLINE  inline
    #define INLINEH inline

#endif

/**********************************************************************************************************************/

typedef struct {
    unsigned int deviceNo;
    unsigned int blockSizeAlign;
    unsigned int blockYSizeAlign;
    unsigned int maxTexYSize;
    unsigned int maxTexXSize;
} _DeviceProps;

/**********************************************************************************************************************/

template<class T> void _ClearHostBuf(T *&buf);

template<class T> void _AllocHostBuf(_Session *s, T *&buf, const T *sBuf, unsigned int numElements, const char *desc);

template<class T> void _DeallocHostBuf(T *&buf);

/**********************************************************************************************************************/

static void _ClaimDevice(_Session *s, int desiredDeviceNo, bool nice, _DeviceProps &props);

static void _ReleaseDevice();

/**********************************************************************************************************************/

static void _ClearTex(_TEXTUREA &ta, _TEXTUREB &tb);

static void _AllocTexLinear(_Session *s, _TEXTUREA &ta, _TEXTUREB &tb, const float *sBuf, unsigned int ySize,
    unsigned int xCount, const char *desc);

static void _AllocTexArray(_Session *s, _TEXTUREA &ta, _TEXTUREB &tb, const float *sBuf, unsigned int ySize,
    unsigned int xCount, const char *desc);

static void _DeallocTex(_TEXTUREA &ta, _TEXTUREB &tb);

static void _PublishTex(_Session *s, _TEXTUREA &ta, _TEXTUREB &tb, const float *sBuf, unsigned int ySize,
    unsigned int xCount, const char *desc);

static void _CopyTex(_Session *s, float *dBuf, unsigned int dHeight, char dest, _TEXTUREA &ta, _TEXTUREB &tb,
    unsigned int yOff, unsigned int xOff, unsigned int yCount, unsigned int xCount, const char *desc);

static void _UpdateTex(_Session *s, _TEXTUREA &ta, _TEXTUREB &tb, unsigned int yOff, unsigned int xOff,
    const float *sBuf, unsigned int sHeight, char src, unsigned int yCount, unsigned int xCount, const char *desc);

/**********************************************************************************************************************/

template<class T> void _ClearBuf(T *&buf);

template<class T> void _AllocBuf(_Session *s, T *&buf, const T *sBuf, unsigned int numElements, const char *desc);

template<class T> void _DeallocBuf(T *&buf);

template<class T> void _CopyBuf1D(_Session *s, T *dBuf, char dest, const T *buf, unsigned int count, const char *desc);

template<class T> void _CopyBuf2D(_Session *s, T *dBuf, unsigned int dHeight, char dest, const T *buf,
    unsigned int height, unsigned int yCount, unsigned int xCount, const char *desc);

template<class T> void _CopyBuf3D(_Session *s, T *dBuf, unsigned int dHeight, unsigned int dWidth, char dest,
    const T *buf, unsigned int height, unsigned int width, unsigned int yOff, unsigned int xOff, unsigned int zOff,
    unsigned int yCount, unsigned int xCount, unsigned int zCount, const char *desc);

template<class T> void _UpdateBuf1D(_Session *s, T *buf, const T *sBuf, char src, unsigned int count, const char *desc);

template<class T> void _UpdateBuf2D(_Session *s, T *buf, unsigned int height, const T *sBuf, unsigned int sHeight,
    char src, unsigned int yCount, unsigned int xCount, const char *desc);

template<class T> void _UpdateBuf3D(_Session *s, T *buf, unsigned int height, unsigned int width, unsigned int yOff,
    unsigned int xOff, unsigned int zOff, const T *sBuf, unsigned int sHeight, unsigned int sWidth, char src,
    unsigned int yCount, unsigned int xCount, unsigned int zCount, const char *desc);

/**********************************************************************************************************************/

template<class T> void _AllocConst(_Session *s, const char *constName, T *cPtr, const T *sPtr,
    unsigned int numElements);

template<class T> void _CopyConst(_Session *s, T *dPtr, const char *constName, const T *cPtr, unsigned int off,
    unsigned int count);

template<class T> void _UpdateConst(_Session *s, const char *constName, T *cPtr, unsigned int off, const T *sPtr,
    unsigned int count);
