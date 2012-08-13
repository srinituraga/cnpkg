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

void _ClaimDevice(_Session *s, int desiredDeviceNo, bool nice, _DeviceProps &props) {

    props.deviceNo = 0;

    // These limits don't really apply to the CPU case, but we abide by them anyway for consistent behavior.
    props.blockSizeAlign  = 32;
    props.blockYSizeAlign = 16;
    props.maxTexYSize     = 1 << 16;
    props.maxTexXSize     = 1 << 15;

}

/**********************************************************************************************************************/

void _ReleaseDevice() {

}

/**********************************************************************************************************************/

void _ClearTex(_TexPtrA &ta, _TexPtrB &tb) {

    ta.linear = false;
    ta.buf    = NULL;
    tb.buf    = NULL;

}

/**********************************************************************************************************************/

void _AllocTexLinear(_Session *s, _TexPtrA &ta, _TexPtrB &tb, const float *sBuf, unsigned int ySize,
    unsigned int xCount, const char *desc) {

    _ClearTex(ta, tb);

    if ((ySize == 0) || (xCount == 0)) return;

    ta.linear = true;

    tb.buf = sBuf;
    tb.h   = ySize;

}

/**********************************************************************************************************************/

void _AllocTexArray(_Session *s, _TexPtrA &ta, _TexPtrB &tb, const float *sBuf, unsigned int ySize,
    unsigned int xCount, const char *desc) {

    _ClearTex(ta, tb);

    if ((ySize == 0) || (xCount == 0)) return;

    ta.buf = (float *)mxMalloc(xCount * ySize * sizeof(float));
    if (ta.buf == NULL) {
        s->Exit2("unable to allocate %s buffer", desc);
    }
    mexMakeMemoryPersistent(ta.buf);

    if (sBuf != NULL) {
        memcpy(ta.buf, sBuf, xCount * ySize * sizeof(float));
    }

    tb.buf = ta.buf;
    tb.h   = ySize;

}

/**********************************************************************************************************************/

void _DeallocTex(_TexPtrA &ta, _TexPtrB &tb) {

    if (!ta.linear) {
        mxFree(ta.buf);
    }

    _ClearTex(ta, tb);

}

/**********************************************************************************************************************/

void _PublishTex(_Session *s, _TexPtrA &ta, _TexPtrB &tb, const float *sBuf, unsigned int ySize, unsigned int xCount,
    const char *desc) {

    if (ta.linear) {
        s->Exit2("_PublishTex not implemented for linear memory textures");
    }

    if ((ySize == 0) || (xCount == 0)) return;

    memcpy(ta.buf, sBuf, xCount * ySize * sizeof(float));

}

/**********************************************************************************************************************/

void _CopyTex(_Session *s, float *dBuf, unsigned int dHeight, char dest, _TexPtrA &ta, _TexPtrB &tb, unsigned int yOff,
    unsigned int xOff, unsigned int yCount, unsigned int xCount, const char *desc) {

    if (ta.linear) {
        s->Exit2("_CopyTex not implemented for linear memory textures");
    }

    if ((yCount == 0) || (xCount == 0)) return;

    const float *buf = tb.buf + xOff * tb.h + yOff;

    for (unsigned int i = 0; i < xCount; i++) {
        memcpy(dBuf, buf, yCount * sizeof(float));
        dBuf += dHeight;
        buf  += tb.h;
    }

}

/**********************************************************************************************************************/

void _UpdateTex(_Session *s, _TexPtrA &ta, _TexPtrB &tb, unsigned int yOff, unsigned int xOff, const float *sBuf,
    unsigned int sHeight, char src, unsigned int yCount, unsigned int xCount, const char *desc) {

    if (ta.linear) {
        s->Exit2("_UpdateTex not implemented for linear memory textures");
    }

    if ((yCount == 0) || (xCount == 0)) return;

    float *buf = ta.buf + xOff * tb.h + yOff;

    for (unsigned int i = 0; i < xCount; i++) {
        memcpy(buf, sBuf, yCount * sizeof(float));
        buf  += tb.h;
        sBuf += sHeight;
    }

}

/**********************************************************************************************************************/

template<class T> void _ClearBuf(T *&buf) {

    _ClearHostBuf(buf);

}

/**********************************************************************************************************************/

template<class T> void _AllocBuf(_Session *s, T *&buf, const T *sBuf, unsigned int numElements, const char *desc) {

    _AllocHostBuf(s, buf, sBuf, numElements, desc);

}

/**********************************************************************************************************************/

template<class T> void _DeallocBuf(T *&buf) {

    _DeallocHostBuf(buf);

}

/**********************************************************************************************************************/

template<class T> void _CopyBuf1D(_Session *s, T *dBuf, char dest, const T *buf, unsigned int count, const char *desc) {

    if (count == 0) return;

    memcpy(dBuf, buf, count * sizeof(T));

}

/**********************************************************************************************************************/

template<class T> void _CopyBuf2D(_Session *s, T *dBuf, unsigned int dHeight, char dest, const T *buf,
    unsigned int height, unsigned int yCount, unsigned int xCount, const char *desc) {

    if ((yCount == 0) || (xCount == 0)) return;

    for (unsigned int i = 0; i < xCount; i++, dBuf += dHeight, buf += height) {
        memcpy(dBuf, buf, yCount * sizeof(T));
    }

}

/**********************************************************************************************************************/

template<class T> void _CopyBuf3D(_Session *s, T *dBuf, unsigned int dHeight, unsigned int dWidth, char dest,
    const T *buf, unsigned int height, unsigned int width, unsigned int yOff, unsigned int xOff, unsigned int zOff,
    unsigned int yCount, unsigned int xCount, unsigned int zCount, const char *desc) {

    if ((yCount == 0) || (xCount == 0) || (zCount == 0)) return;

    buf += (zOff * width + xOff) * height + yOff;

    for (unsigned int j = 0; j < zCount; j++) {
        for (unsigned int i = 0; i < xCount; i++) {
            memcpy(dBuf, buf, yCount * sizeof(T));
            dBuf += dHeight;
            buf  += height;
        }
        dBuf += dHeight * (dWidth - xCount);
        buf  += height  * (width  - xCount);
    }

}

/**********************************************************************************************************************/

template<class T> void _UpdateBuf1D(_Session *s, T *buf, const T *sBuf, char src, unsigned int count,
    const char *desc) {

    if (count == 0) return;

    memcpy(buf, sBuf, count * sizeof(T));

}

/**********************************************************************************************************************/

template<class T> void _UpdateBuf2D(_Session *s, T *buf, unsigned int height, const T *sBuf, unsigned int sHeight,
    char src, unsigned int yCount, unsigned int xCount, const char *desc) {

    if ((yCount == 0) || (xCount == 0)) return;

    for (unsigned int i = 0; i < xCount; i++, buf += height, sBuf += sHeight) {
        memcpy(buf, sBuf, yCount * sizeof(T));
    }

}

/**********************************************************************************************************************/

template<class T> void _UpdateBuf3D(_Session *s, T *buf, unsigned int height, unsigned int width, unsigned int yOff,
    unsigned int xOff, unsigned int zOff, const T *sBuf, unsigned int sHeight, unsigned int sWidth, char src,
    unsigned int yCount, unsigned int xCount, unsigned int zCount, const char *desc) {

    if ((yCount == 0) || (xCount == 0) || (zCount == 0)) return;

    buf += (zOff * width + xOff) * height + yOff;

    for (unsigned int j = 0; j < zCount; j++) {
        for (unsigned int i = 0; i < xCount; i++) {
            memcpy(buf, sBuf, yCount * sizeof(T));
            buf  += height;
            sBuf += sHeight;
        }
        buf  += height  * (width  - xCount);
        sBuf += sHeight * (sWidth - xCount);
    }

}

/**********************************************************************************************************************/

template<class T> void _AllocConst(_Session *s, const char *constName, T *cPtr, const T *sPtr,
    unsigned int numElements) {

    if (numElements == 0) return;

    memcpy(cPtr, sPtr, numElements * sizeof(T));

}

/**********************************************************************************************************************/

template<class T> void _CopyConst(_Session *s, T *dPtr, const char *constName, const T *cPtr, unsigned int off,
    unsigned int count) {

    if (count == 0) return;

    memcpy(dPtr, cPtr + off, count * sizeof(T));

}

/**********************************************************************************************************************/

template<class T> void _UpdateConst(_Session *s, const char *constName, T *cPtr, unsigned int off, const T *sPtr,
    unsigned int count) {

    if (count == 0) return;

    memcpy(cPtr + off, sPtr, count * sizeof(T));

}
