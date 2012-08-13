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

    __global__ void _USER_KERNEL_NAME(

        unsigned int      _mvOff,
        unsigned int      _cOff,
        const float      *_dData,
        const float      *_dWData,
        _OutTable         _tOut,
        unsigned int      _iterNo,

        const ushort8    *_bPtr,
        unsigned int      _bTotal,
        unsigned int      _bStart,
        unsigned int      _bCount,
        const _LayerData *_dLayers

    ) {

        unsigned int _bid = blockIdx.y * gridDim.x + blockIdx.x;
        if (_bid >= _bCount) return;

        const unsigned int _tid = threadIdx.y * blockDim.x + threadIdx.x;

        __shared__ unsigned int _xStart;
        __shared__ unsigned int _yStart;
        __shared__ unsigned int _z;
        __shared__ unsigned int _blockYSize;
        __shared__ unsigned int _txCount;
        __shared__ unsigned int _tyCount;
        __shared__ unsigned int _txStep;
        __shared__ unsigned int _tyStep;
        if (_tid == 0) {
            _bid += _bStart;
            if (_bid >= _bTotal) _bid -= _bTotal;
            const ushort8 _bd = _bPtr[_bid];
            _xStart     = _bd.x;
            _yStart     = _bd.y;
            _z          = _bd.z;
            _blockYSize = _bd.w;
            _txCount    = _bd.a;
            _tyCount    = _bd.b;
            _txStep     = _bd.c;
            _tyStep     = _bd.d;
        }
        __syncthreads();

        __shared__ _LayerData _p;
        if (_tid < _LAYERDATA_UINTS) {
            _p.m_array[_tid] = _dLayers[_z].m_array[_tid];
        }
        __syncthreads();

        const unsigned int _tx = _tid / _blockYSize;
        const unsigned int _ty = _tid - _tx * _blockYSize;
        if ((_tx >= _txCount) || (_ty >= _tyCount)) return;
        unsigned int _x = _xStart + _tx * _txStep;
        unsigned int _y = _yStart + _ty * _tyStep;

        int _sc = -1;

        #if defined(_USES_INT32_ARRAYS) || defined(_USES_FLOAT_ARRAYS) || defined(_USES_DOUBLE_ARRAYS)
            const unsigned int _blockSize = blockDim.x * blockDim.y;
        #endif
        #if defined(_USES_INT32_ARRAYS)
            extern __shared__ int _int32Arrays[];
        #endif
        #if defined(_USES_FLOAT_ARRAYS)
            extern __shared__ float _floatArrays[];
        #endif
        #if defined(_USES_DOUBLE_ARRAYS)
            extern __shared__ double _doubleArrays[];
        #endif

        #include _USER_KERNEL_DEF

    }

#else

    // We put everything in an inline function so that "return" inside the function body will only end computation for
    // a single cell, as in the GPU case.

    inline void _USER_KERNEL_NAME2(

        unsigned int          _mvOff,
        unsigned int          _cOff,
        const float          *_dData,
        const float          *_dWData,
        _OutTable            &_tOut,
        unsigned int          _iterNo,

        _Session             *_session, 
        unsigned int          _z,
        const _LayerData     &_p,
        const float          *_cData,
        const unsigned short *_cMeta,
        unsigned int          _y,
        unsigned int          _x

    ) {

        int _sc = -1;

        #if defined(_USES_INT32_ARRAYS)
            int _int32Arrays[_INT32_ARRAY_SIZE];
        #endif
        #if defined(_USES_FLOAT_ARRAYS)
            float _floatArrays[_FLOAT_ARRAY_SIZE];
        #endif
        #if defined(_USES_DOUBLE_ARRAYS)
            double _doubleArrays[_DOUBLE_ARRAY_SIZE];
        #endif

        #include _USER_KERNEL_DEF

    }

    void _USER_KERNEL_NAME(

        unsigned int          _mvOff,
        unsigned int          _cOff,
        const float          *_dData,
        const float          *_dWData,
        _OutTable             _tOut,
        unsigned int          _iterNo,

        _Session             *_session, 
        unsigned int          _z,
        const _LayerData     &_p,
        const float          *_cData,
        const unsigned short *_cMeta,
        unsigned int          _yStart,
        unsigned int          _yCount,
        unsigned int          _yStep,
        unsigned int          _xStart,
        unsigned int          _xCount,
        unsigned int          _xStep

    ) {

        for (unsigned int j = 0, x = _xStart; j < _xCount; j++, x += _xStep) {
        for (unsigned int i = 0, y = _yStart; i < _yCount; i++, y += _yStep) {

            _USER_KERNEL_NAME2(

                _mvOff,
                _cOff,
                _dData,
                _dWData,
                _tOut,
                _iterNo,

                _session, 
                _z,
                _p,
                _cData,
                _cMeta,
                y,
                x

                );

        }
        }

    }

#endif
