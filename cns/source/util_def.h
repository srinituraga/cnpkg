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

template<class T> void _ClearHostBuf(T *&buf) {

    buf = NULL;

}

/**********************************************************************************************************************/

template<class T> void _AllocHostBuf(_Session *s, T *&buf, const T *sBuf, unsigned int numElements, const char *desc) {

    if (numElements == 0) {
        buf = NULL;
        return;
    }

    buf = (T *)mxMalloc(numElements * sizeof(T));
    if (buf == NULL) {
        s->Exit("unable to allocate %s buffer", desc);
    }
    mexMakeMemoryPersistent(buf);

    if (sBuf != NULL) {
        memcpy(buf, sBuf, numElements * sizeof(T));
    }

}

/**********************************************************************************************************************/

template<class T> void _DeallocHostBuf(T *&buf) {

    if (buf != NULL) {
        mxFree(buf);
        buf = NULL;
    }

}

/**********************************************************************************************************************/

#ifdef _GPU
    #include "util_def_cuda.h"
#else
    #include "util_def_cpu.h"
#endif
