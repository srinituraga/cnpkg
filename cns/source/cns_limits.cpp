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

#include "common_dec.h"

/**********************************************************************************************************************/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    unsigned int mode = (unsigned int)mxGetScalar(prhs[0]);

    double val;
    switch (mode) {
    case 0: val = (double)CNS_INTMIN; break;
    case 1: val = (double)CNS_INTMAX; break;
    case 2: val = (double)CNS_FLTMIN; break;
    case 3: val = (double)CNS_FLTMAX; break;
    }

    plhs[0] = mxCreateDoubleScalar(val);

}
