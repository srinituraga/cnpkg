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

unsigned int _GetDimSize(const mxArray *array, unsigned int d1, unsigned int nDims) {

    unsigned int d2 = d1 + nDims - 1;

    unsigned int size = 1;

    for (unsigned int d = d1; (d <= d2) && (d < (unsigned int)mxGetNumberOfDimensions(array)); d++) {
        size *= mxGetDimensions(array)[d];
    }

    return size;

}

/**********************************************************************************************************************/

char *_CBYX2S(const mxArray *cb, unsigned int z, unsigned int y, unsigned int x, unsigned int roll, bool internal,
    char *buf) {

    mxArray *rhs[7];
    mxArray *lhs[1];

    rhs[0] = (mxArray *)cb;
    rhs[1] = mxCreateString("CBYX2S");
    rhs[2] = mxCreateDoubleScalar(z + 1);
    rhs[3] = mxCreateDoubleScalar(y + 1);
    rhs[4] = mxCreateDoubleScalar(x + 1);
    rhs[5] = mxCreateDoubleScalar(roll);
    rhs[6] = mxCreateLogicalScalar(internal);

    mexCallMATLAB(1, lhs, 7, rhs, "feval");

    mxGetString(lhs[0], buf, mxGetNumberOfElements(lhs[0]) + 1); // assume buffer is big enough

    for (unsigned int i = 1; i <= 6; i++) mxDestroyArray(rhs[i]);
    mxDestroyArray(lhs[0]);

    return buf;

}

/**********************************************************************************************************************/

void *operator new(size_t size) {

    void *ptr = mxMalloc(size);

    mexMakeMemoryPersistent(ptr);

    return ptr;

}

/**********************************************************************************************************************/

void *operator new[](size_t size) {

    void *ptr = mxMalloc(size);

    mexMakeMemoryPersistent(ptr);

    return ptr;

}

/**********************************************************************************************************************/

void operator delete(void *ptr) {

    mxFree(ptr);

}

/**********************************************************************************************************************/

void operator delete[](void *ptr) {

    mxFree(ptr);

}

/**********************************************************************************************************************/

float _IntAsFloat(int a) {

    union {int i; float f;} u;

    u.i = a;

    return u.f;

}

/**********************************************************************************************************************/

int _FloatAsInt(float a) {

    union {int i; float f;} u;

    u.f = a;

    return u.i;

}
