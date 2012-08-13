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

#include "mex.h"
#include <stdarg.h>
#include <string.h>
#include <float.h>

/**********************************************************************************************************************/

#if (SHRT_MAX != 32767) || (INT_MAX != 2147483647)
    #error unsupported size for short or int
#endif

/**********************************************************************************************************************/

#define CNS_INTMIN INT_MIN
#define CNS_INTMAX INT_MAX
#define CNS_FLTMIN (-FLT_MAX)
#define CNS_FLTMAX FLT_MAX

// These are fairly arbitrary limits governing the size of statically allocated arrays in host memory.  There would not
// be much penalty for increasing them, and they could be removed altogether if we were willing to make the code a bit
// more complicated (i.e. use dynamic allocation and deallocation).

const unsigned int _MAX_LAYERS = 256;
const unsigned int _MAX_DIMS   = 10;
const unsigned int _ERRMSG_LEN = 512;

/**********************************************************************************************************************/

static unsigned int _GetDimSize(const mxArray *array, unsigned int d1, unsigned int nDims = 1);

/**********************************************************************************************************************/

static char *_CBYX2S(const mxArray *cb, unsigned int z, unsigned int y, unsigned int x, unsigned int roll,
    bool internal, char *buf);

/**********************************************************************************************************************/

void *operator new(size_t size);
void *operator new[](size_t size);
void operator delete(void *ptr);
void operator delete[](void *ptr);

/**********************************************************************************************************************/

inline float _IntAsFloat(int a);
inline int _FloatAsInt(float a);
