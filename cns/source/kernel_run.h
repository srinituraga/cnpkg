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

    case _USER_KERNEL_TYPENO:

        _USER_KERNEL_NAME<<<gridDim, blockDim, m_blockSize * _USER_KERNEL_ARRAYBYTES>>>(

            m_s->GetMMVOff(),
            m_s->GetMCOff(),
            m_s->GetDData(0),
            m_s->GetDWData(0),
            m_s->GetTOut(),
            m_s->GetIterNo(),
            m_phase,

            m_bPtr,
            m_bCount,
            bStart,
            bCount,
            m_s->GetDLayers(0)

            );

        // First, check to see if the kernel failed to launch at all.

        if ((err = cudaGetLastError()) != cudaSuccess) {
            _CudaExit(m_s, err, "unable to launch kernel (type '%s')", m_type);
        }

        // Kernel calls return immediately after launching.  If you made a bunch in a row (without synchronizing),
        // CUDA would queue them, but only one kernel at a time would actually run on the device.  We synchronize
        // after every call.  This allows us to retrieve the error code if the kernel fails after launch.

        if ((err = cudaThreadSynchronize()) != cudaSuccess) {
            _CudaExit(m_s, err, "kernel failed after launch (type '%s')", m_type);
        }

        break;

#else

    case _USER_KERNEL_TYPENO:

        _USER_KERNEL_NAME(

            m_s->GetMMVOff(),
            m_s->GetMCOff(),
            m_s->GetDData(0),
            m_s->GetDWData(0),
            m_s->GetTOut(),
            m_s->GetIterNo(),
            phase,

            m_s,
            m_z,
            *(_LayerData *)this,
            m_s->GetCData(),
            m_s->GetCMeta(),
            xStart,
            xCount

            );

        break;

#endif
