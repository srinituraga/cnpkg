/***********************************************************************************************************************
*
* Copyright (C) 2010 by Jim Mutch (www.jimmutch.com).
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

/**********************************************************************************************************************/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    int count;
    cudaGetDeviceCount(&count);

    if (count == 1) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        if ((props.major == 9999) && (props.minor == 9999)) count = 0;
    }

    plhs[0] = mxCreateDoubleScalar((double)count);

}
