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

    );

#else

    static void _USER_KERNEL_NAME(

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

    );

#endif
