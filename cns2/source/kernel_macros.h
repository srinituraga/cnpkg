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

    #define _G_CDATA                 _G_CDATASYM
    #define _G_CMETA                 _G_CMETASYM

    #define _MMVOff(M,N,E,L)         (_G_CMETA[_mvOff                 + M] + (E) * L)
    #define _GMVOff(M,N,E,L)         (_G_CMETA[_p.m_gmvOff            + M] + (E) * L)
    #define _LMVOff(M,N,E,L)         (_G_CMETA[_p.m_mvOff             + M] + (E) * L)
    #define _ZGMVOff(Z,M,N,E,L)      (_G_CMETA[_LTValue(Z,_LT_GMVOFF) + M] + (E) * L)
    #define _ZLMVOff(Z,M,N,E,L)      (_G_CMETA[_LTValue(Z,_LT_MVOFF)  + M] + (E) * L)

    #define _ERROR(...)              do {} while(false)
    #define _PRINT(...)              do {} while(false)

    #define _Int32ArrayElement(F,E)  _int32Arrays [(F + (E)) * _blockSize + _tid]
    #define _FloatArrayElement(F,E)  _floatArrays [(F + (E)) * _blockSize + _tid]
    #define _DoubleArrayElement(F,E) _doubleArrays[(F + (E)) * _blockSize + _tid]

    #define _SELECT_SYN(E) \
        const unsigned int _si = (E); \
        const ushort4      _sm = _p.m_smPtr[(_si * _p.m_xCount + _x) * _p.m_ySize + _y];

    #define _SESSION

#else

    #define _G_CDATA                 _cData
    #define _G_CMETA                 _cMeta

    #define _MMVOff(M,N,E,L)         _MVOff(_session, _G_CMETA + _mvOff                 + M, #N, E, L, _z, _y, _x)
    #define _GMVOff(M,N,E,L)         _MVOff(_session, _G_CMETA + _p.m_gmvOff            + M, #N, E, L, _z, _y, _x)
    #define _LMVOff(M,N,E,L)         _MVOff(_session, _G_CMETA + _p.m_mvOff             + M, #N, E, L, _z, _y, _x)
    #define _ZGMVOff(Z,M,N,E,L)      _MVOff(_session, _G_CMETA + _LTValue(Z,_LT_GMVOFF) + M, #N, E, L, _z, _y, _x)
    #define _ZLMVOff(Z,M,N,E,L)      _MVOff(_session, _G_CMETA + _LTValue(Z,_LT_MVOFF)  + M, #N, E, L, _z, _y, _x)

    #define _ERROR(...)              _session->NeuronExit(_z, _y, _x, __VA_ARGS__)
    #define _PRINT(...)              _session->NeuronInfo(_z, _y, _x, __VA_ARGS__)

    #define _Int32ArrayElement(F,E)  _int32Arrays [F + (E)]
    #define _FloatArrayElement(F,E)  _floatArrays [F + (E)]
    #define _DoubleArrayElement(F,E) _doubleArrays[F + (E)]

    #define _SELECT_SYN(E) \
        const unsigned int _si = _CheckSyn(_session, _NUM_SYN, E, _z, _y, _x); \
        const ushort4      _sm = _p.m_smPtr[(_si * _p.m_xCount + _x) * _p.m_ySize + _y];

    inline unsigned int _CheckSyn(_Session *s, unsigned int count, unsigned int e, unsigned int z, unsigned int y,
        unsigned int x) {
        if (e >= count) {
            s->NeuronExit(z, y, x, "requested synapse number (%u) exceeds number of synapses (%u)", e, count);
        }
        return e;
    }

    inline unsigned int _MVOff(_Session *s, const unsigned short *meta, const char *n, unsigned int e, unsigned int len,
        unsigned int z, unsigned int y, unsigned int x) {
        if (e >= meta[1]) {
            s->NeuronExit(z, y, x, "requested value number (%u) exceeds number of values (%u) for field '%s'",
                e, meta[1], n);
        }
        return meta[0] + e * len;
    }

    inline int   min  (int   a, int   b) { return (a <= b) ? a : b; }
    inline int   max  (int   a, int   b) { return (a >= b) ? a : b; }
    inline float fminf(float a, float b) { return (a <= b) ? a : b; }
    inline float fmaxf(float a, float b) { return (a >= b) ? a : b; }

    #define __int_as_float _IntAsFloat
    #define __float_as_int _FloatAsInt

    #define _SESSION _session,

#endif

#define _THIS_Z                    ((int)_z)

#define _NUM_SYN                   (_sc >= 0 ? _sc : (_sc = _p.m_nmPtr[_x * _p.m_ySize + _y]))
#define _SYN_Z                     ((int)_sm.z)
#define _SYN_TYPE                  ((int)_sm.w)

#define _LTPtr(Z,A)                (_G_CMETA + Z * _LT_LEN + A)
#define _LTValue(Z,A)              _G_CMETA[Z * _LT_LEN + A]

#define _MMVCount(M)               ((int)_G_CMETA[_mvOff                 + M + 1])
#define _GMVCount(M)               ((int)_G_CMETA[_p.m_gmvOff            + M + 1])
#define _LMVCount(M)               ((int)_G_CMETA[_p.m_mvOff             + M + 1])
#define _ZGMVCount(Z,M)            ((int)_G_CMETA[_LTValue(Z,_LT_GMVOFF) + M + 1])
#define _ZLMVCount(Z,M)            ((int)_G_CMETA[_LTValue(Z,_LT_MVOFF)  + M + 1])

#define _GetLayerSz(S,D)           _LTrans##S##_s##D(_LTPtr(_z,_LT_SIZ2P))
#define _GetZLayerSz(Z,S,D)        _LTrans##S##_s##D(_LTPtr(Z,_LT_SIZ2P))

#define _GetCoords(S)              _LTrans##S##_ic(_LTPtr(_z,_LT_SIZ2P), _y, _x, _ALLCOORDS)

#define _SetCoords(K,S,...)        _KTrans##K##s##S##_sc(_LTPtr(_z,_LT_SIZ2P), _y, _x, _sc, _ALLCOORDS, __VA_ARGS__)

#define _GetPCoord(S,D)            _LTrans##S##_c##D(_LTPtr(_sm.z,_LT_SIZ2P), _sm.y, _sm.x)

#define _TPtr(F)                   (_G_CMETA + _p.m_tOff            + F * 2)
#define _ZTPtr(Z,F)                (_G_CMETA + _LTValue(Z,_LT_TOFF) + F * 2)

#define _GetCConst(B,F)            _CTrans##B##_ln(_SESSION _TPtr(F), _y, _x)
#define _GetCVar(B,F)              _VTrans##B##_ln(_SESSION _TPtr(F), _y, _x)
#define _GetPCConst(B,F)           _CTrans##B##_ln(_SESSION _ZTPtr(_sm.z,F), _sm.y, _sm.x)
#define _GetPCVar(B,F)             _VTrans##B##_ln(_SESSION _ZTPtr(_sm.z,F), _sm.y, _sm.x)
#define _GetZCConst(Z,B,F,...)     _CTrans##B##_lk(_SESSION _ZTPtr(Z,F), _LTPtr(Z,_LT_SIZ2P), __VA_ARGS__)
#define _GetZCVar(Z,B,F,...)       _VTrans##B##_lk(_SESSION _ZTPtr(Z,F), _LTPtr(Z,_LT_SIZ2P), __VA_ARGS__)
#define _GetZCConst1(Z,B,F,...)    _CTrans##B##_gc(_ZTPtr(Z,F), _LTPtr(Z,_LT_SIZ2P), __VA_ARGS__)
#define _GetZCVar1(Z,B,F,...)      _VTrans##B##_gc(_ZTPtr(Z,F), _LTPtr(Z,_LT_SIZ2P), __VA_ARGS__)
#define _SetCVar(B,R,F,V)          _VTrans##B##_wn(_tOut.ptr[R], _tOut.h[R], _TPtr(F), _y, _x, V)

#define _DefCConstH(B)             _CTrans##B##_h
#define _DefCVarH(B)               _VTrans##B##_h
#define _GetZCConstH(Z,B,F)        _CTrans##B##_mh(_ZTPtr(Z,F), _LTPtr(Z,_LT_SIZ2P))
#define _GetZCVarH(Z,B,F)          _VTrans##B##_mh(_ZTPtr(Z,F), _LTPtr(Z,_LT_SIZ2P))
#define _GetHCConstSz(H,B,D)       _CTrans##B##_sh##D(H)
#define _GetHCVarSz(H,B,D)         _VTrans##B##_sh##D(H)
#define _GetHCConst(H,B,...)       _CTrans##B##_lkh(_SESSION H, __VA_ARGS__)
#define _GetHCVar(H,B,...)         _VTrans##B##_lkh(_SESSION H, __VA_ARGS__)
#define _GetHCConst1(H,B,...)      _CTrans##B##_gch(H, __VA_ARGS__)
#define _GetHCVar1(H,B,...)        _VTrans##B##_gch(H, __VA_ARGS__)

#define _GetCConst2(B,Y,X)         _CTrans##B##_lc(_SESSION Y, X)
#define _GetCVar2(B,Y,X)           _VTrans##B##_lc(_SESSION Y, X)

#define _DefArrayH(R)              _ATrans##R##_h
#define _DefTexH(R)                _TTrans##R##_h
#define _GetHArray(R,H,...)        _ATrans##R##_lkh(_dData,  H, __VA_ARGS__)
#define _GetHTex(R,H,...)          _TTrans##R##_lkh(_SESSION H, __VA_ARGS__)
#define _GetHArray1(R,H,...)       _ATrans##R##_gch(H, __VA_ARGS__)
#define _GetHTex1(R,H,...)         _TTrans##R##_gch(H, __VA_ARGS__)
#define _GetHArraySz(R,H,D)        _ATrans##R##_sh##D(H)
#define _GetHTexSz(R,H,D)          _TTrans##R##_sh##D(H)

#define _GetMConst(F)              _G_CDATA[_cOff + F       ]
#define _GetMConstMV(M,N,E)        _G_CDATA[_MMVOff(M,N,E,1)]
#define _GetMArrayH(R,M,N,E)       _ATrans##R##_mh(_G_CMETA + _MMVOff(M,N,E,_ALEN_##R))
#define _GetMTexH(R,M,N,E)         _TTrans##R##_mh(_G_CMETA + _MMVOff(M,N,E,_TLEN_##R))
#define _GetMArray(R,M,N,E,...)    _ATrans##R##_lk(_dData,  _G_CMETA + _MMVOff(M,N,E,_ALEN_##R), __VA_ARGS__)
#define _GetMTex(R,M,N,E,...)      _TTrans##R##_lk(_SESSION _G_CMETA + _MMVOff(M,N,E,_TLEN_##R), __VA_ARGS__)
#define _GetMTex1(R,M,N,E,...)     _TTrans##R##_gc(_G_CMETA + _MMVOff(M,N,E,_TLEN_##R), __VA_ARGS__)
#define _GetMArraySz(R,M,N,E,D)    _ATrans##R##_s##D(_G_CMETA + _MMVOff(M,N,E,_ALEN_##R))
#define _GetMTexSz(R,M,N,E,D)      _TTrans##R##_s##D(_G_CMETA + _MMVOff(M,N,E,_TLEN_##R))

#define _GetGConst(F)              _G_CDATA[_p.m_gcOff + F  ]
#define _GetGConstMV(M,N,E)        _G_CDATA[_GMVOff(M,N,E,1)]
#define _GetGArrayH(R,M,N,E)       _ATrans##R##_mh(_G_CMETA + _GMVOff(M,N,E,_ALEN_##R))
#define _GetGTexH(R,M,N,E)         _TTrans##R##_mh(_G_CMETA + _GMVOff(M,N,E,_TLEN_##R))
#define _GetGArray(R,M,N,E,...)    _ATrans##R##_lk(_dData,  _G_CMETA + _GMVOff(M,N,E,_ALEN_##R), __VA_ARGS__)
#define _GetGTex(R,M,N,E,...)      _TTrans##R##_lk(_SESSION _G_CMETA + _GMVOff(M,N,E,_TLEN_##R), __VA_ARGS__)
#define _GetGTex1(R,M,N,E,...)     _TTrans##R##_gc(_G_CMETA + _GMVOff(M,N,E,_TLEN_##R), __VA_ARGS__)
#define _GetGArraySz(R,M,N,E,D)    _ATrans##R##_s##D(_G_CMETA + _GMVOff(M,N,E,_ALEN_##R))
#define _GetGTexSz(R,M,N,E,D)      _TTrans##R##_s##D(_G_CMETA + _GMVOff(M,N,E,_TLEN_##R))

#define _GetLConst(F)              _G_CDATA[_p.m_cOff + F   ]
#define _GetLConstMV(M,N,E)        _G_CDATA[_LMVOff(M,N,E,1)]
#define _GetLArrayH(R,M,N,E)       _ATrans##R##_mh(_G_CMETA + _LMVOff(M,N,E,_ALEN_##R))
#define _GetLTexH(R,M,N,E)         _TTrans##R##_mh(_G_CMETA + _LMVOff(M,N,E,_TLEN_##R))
#define _GetLArray(R,M,N,E,...)    _ATrans##R##_lk(_dData,  _G_CMETA + _LMVOff(M,N,E,_ALEN_##R), __VA_ARGS__)
#define _GetLTex(R,M,N,E,...)      _TTrans##R##_lk(_SESSION _G_CMETA + _LMVOff(M,N,E,_TLEN_##R), __VA_ARGS__)
#define _GetLTex1(R,M,N,E,...)     _TTrans##R##_gc(_G_CMETA + _LMVOff(M,N,E,_TLEN_##R), __VA_ARGS__)
#define _GetLArraySz(R,M,N,E,D)    _ATrans##R##_s##D(_G_CMETA + _LMVOff(M,N,E,_ALEN_##R))
#define _GetLTexSz(R,M,N,E,D)      _TTrans##R##_s##D(_G_CMETA + _LMVOff(M,N,E,_TLEN_##R))

#define _GetZGConst(Z,F)           _G_CDATA[_LTValue(Z,_LT_GCOFF) + F]
#define _GetZGConstMV(Z,M,N,E)     _G_CDATA[_ZGMVOff(Z,M,N,E,1)      ]
#define _GetZGArrayH(Z,R,M,N,E)    _ATrans##R##_mh(_G_CMETA + _ZGMVOff(Z,M,N,E,_ALEN_##R))
#define _GetZGTexH(Z,R,M,N,E)      _TTrans##R##_mh(_G_CMETA + _ZGMVOff(Z,M,N,E,_TLEN_##R))
#define _GetZGArray(Z,R,M,N,E,...) _ATrans##R##_lk(_dData,  _G_CMETA + _ZGMVOff(Z,M,N,E,_ALEN_##R), __VA_ARGS__)
#define _GetZGTex(Z,R,M,N,E,...)   _TTrans##R##_lk(_SESSION _G_CMETA + _ZGMVOff(Z,M,N,E,_TLEN_##R), __VA_ARGS__)
#define _GetZGTex1(Z,R,M,N,E,...)  _TTrans##R##_gc(_G_CMETA + _ZGMVOff(Z,M,N,E,_TLEN_##R), __VA_ARGS__)
#define _GetZGArraySz(Z,R,M,N,E,D) _ATrans##R##_s##D(_G_CMETA + _ZGMVOff(Z,M,N,E,_ALEN_##R))
#define _GetZGTexSz(Z,R,M,N,E,D)   _TTrans##R##_s##D(_G_CMETA + _ZGMVOff(Z,M,N,E,_TLEN_##R))

#define _GetZLConst(Z,F)           _G_CDATA[_LTValue(Z,_LT_COFF) + F]
#define _GetZLConstMV(Z,M,N,E)     _G_CDATA[_ZLMVOff(Z,M,N,E,1)     ]
#define _GetZLArrayH(Z,R,M,N,E)    _ATrans##R##_mh(_G_CMETA + _ZLMVOff(Z,M,N,E,_ALEN_##R))
#define _GetZLTexH(Z,R,M,N,E)      _TTrans##R##_mh(_G_CMETA + _ZLMVOff(Z,M,N,E,_TLEN_##R))
#define _GetZLArray(Z,R,M,N,E,...) _ATrans##R##_lk(_dData,  _G_CMETA + _ZLMVOff(Z,M,N,E,_ALEN_##R), __VA_ARGS__)
#define _GetZLTex(Z,R,M,N,E,...)   _TTrans##R##_lk(_SESSION _G_CMETA + _ZLMVOff(Z,M,N,E,_TLEN_##R), __VA_ARGS__)
#define _GetZLTex1(Z,R,M,N,E,...)  _TTrans##R##_gc(_G_CMETA + _ZLMVOff(Z,M,N,E,_TLEN_##R), __VA_ARGS__)
#define _GetZLArraySz(Z,R,M,N,E,D) _ATrans##R##_s##D(_G_CMETA + _ZLMVOff(Z,M,N,E,_ALEN_##R))
#define _GetZLTexSz(Z,R,M,N,E,D)   _TTrans##R##_s##D(_G_CMETA + _ZLMVOff(Z,M,N,E,_TLEN_##R))

#define _GetArray2(R,H,Y,X)        _ATrans##R##_lc(_dData, H, Y, X)
#define _GetTex2(R,Y,X)            _TTrans##R##_lc(_SESSION   Y, X)

#define _GetNField(F)              _p.m_ndPtr[(F                * _p.m_xCount + _x) * _p.m_ySize + _y]
#define _SetNField(F,V)           (_p.m_ndPtr[(F                * _p.m_xCount + _x) * _p.m_ySize + _y] = (V))
#define _GetNFieldMV(M,N,E)        _p.m_ndPtr[(_LMVOff(M,N,E,1) * _p.m_xCount + _x) * _p.m_ySize + _y]
#define _SetNFieldMV(M,N,E,V)     (_p.m_ndPtr[(_LMVOff(M,N,E,1) * _p.m_xCount + _x) * _p.m_ySize + _y] = (V))

#define _GetPNField(B)             _NTrans##B##_ln(_dData, _LTPtr(_sm.z,_LT_SIZ2P), _sm.y, _sm.x)
#define _GetZNField(Z,B,...)       _NTrans##B##_lk(_dData, _LTPtr(Z,_LT_SIZ2P), __VA_ARGS__)
#define _DefNFieldH(B)             _NTrans##B##_h
#define _GetZNFieldH(Z,B)          _NTrans##B##_mh(_LTPtr(Z,_LT_SIZ2P))
#define _GetHNFieldSz(H,B,D)       _NTrans##B##_sh##D(H)
#define _GetHNField(H,B,...)       _NTrans##B##_lkh(_dData, H, __VA_ARGS__)
#define _GetHNField1(H,B,...)      _NTrans##B##_gch(H, __VA_ARGS__)
#define _GetNField2(H,B,Y,X)       _NTrans##B##_lc(_dData, H, Y, X)

#define _GetWField(F)              _p.m_nwPtr[(F * _p.m_xCount + _x) * _p.m_ySize + _y]
#define _SetWField(F,V)           (_p.m_nwOut[(F * _p.m_xCount + _x) * _p.m_ySize + _y] = (V))

#define _GetPWField(B)             _WTrans##B##_ln(_dWData, _LTPtr(_sm.z,_LT_SIZ2P), _sm.y, _sm.x)
#define _GetZWField(Z,B,...)       _WTrans##B##_lk(_dWData, _LTPtr(Z,_LT_SIZ2P), __VA_ARGS__)
#define _DefWFieldH(B)             _WTrans##B##_h
#define _GetZWFieldH(Z,B)          _WTrans##B##_mh(_LTPtr(Z,_LT_SIZ2P))
#define _GetHWFieldSz(H,B,D)       _WTrans##B##_sh##D(H)
#define _GetHWField(H,B,...)       _WTrans##B##_lkh(_dWData, H, __VA_ARGS__)
#define _GetHWField1(H,B,...)      _WTrans##B##_gch(H, __VA_ARGS__)
#define _GetWField2(H,B,Y,X)       _WTrans##B##_lc(_dWData, H, Y, X)

#define _GetSField(F)              _p.m_sdPtr[((F                * _p.m_sSize + _si) * _p.m_xCount + _x) * _p.m_ySize + _y]
#define _SetSField(F,V)           (_p.m_sdPtr[((F                * _p.m_sSize + _si) * _p.m_xCount + _x) * _p.m_ySize + _y] = (V))
#define _GetSFieldMV(M,N,E)        _p.m_sdPtr[((_LMVOff(M,N,E,1) * _p.m_sSize + _si) * _p.m_xCount + _x) * _p.m_ySize + _y]
#define _SetSFieldMV(M,N,E,V)     (_p.m_sdPtr[((_LMVOff(M,N,E,1) * _p.m_sSize + _si) * _p.m_xCount + _x) * _p.m_ySize + _y] = (V))

#define _ITER_NO                   ((int)_iterNo)

/**********************************************************************************************************************/

#define _GetCenter(D,F)    (_GetLConst(F)    + (float)_c##D * _GetLConst(F+1)   )
#define _GetZCenter(Z,F,C) (_GetZLConst(Z,F) + (float)(C)   * _GetZLConst(Z,F+1))

#define _GetRFDistAt(Z,S,D,F,P,R,...) \
    _RFDist(_GetZLayerSz(Z,S,D), _GetZLConst(Z,F), _GetZLConst(Z,F+1), P, R, __VA_ARGS__)

#define _GetRFNearAt(Z,S,D,F,P,N,...) \
    _RFNear(_GetZLayerSz(Z,S,D), _GetZLConst(Z,F), _GetZLConst(Z,F+1), P, N, __VA_ARGS__)

#define _GetRFDist(Z,D1,F1,S2,D2,F2,R,...) _GetRFDistAt(Z, S2, D2, F2, _GetCenter(D1,F1), R, __VA_ARGS__)
#define _GetRFNear(Z,D1,F1,S2,D2,F2,N,...) _GetRFNearAt(Z, S2, D2, F2, _GetCenter(D1,F1), N, __VA_ARGS__)

#define _GetCoordNew(S,D,F) (_c##D - (_GetLayerSz(S,D) - __float_as_int(_GetLConst(F))))

/*--------------------------------------------------------------------------------------------------------------------*/

INLINE bool _RFDist(int t, float s, float d, float c, float r, int &v1, int &v2, int &c1, int &c2) {

    float dd = 1.0f / d;

    c1 = (int)ceilf ((c - r - s) * dd - 0.001f);
    c2 = (int)floorf((c + r - s) * dd + 0.001f);

    v1 = min(max(c1,  0), t    );
    v2 = min(max(c2, -1), t - 1);

    return (v1 <= v2);

}

/*--------------------------------------------------------------------------------------------------------------------*/

INLINE bool _RFDist(int t, float s, float d, float c, float r, int &v1, int &v2) {

    int c1, c2;
    _RFDist(t, s, d, c, r, v1, v2, c1, c2);

    return (v1 == c1) && (v2 == c2);

}

/*--------------------------------------------------------------------------------------------------------------------*/

INLINE bool _RFNear(int t, float s, float d, float c, int n, int &v1, int &v2, int &c1, int &c2) {

    float dd = 1.0f / d;

    c1 = (int)ceilf((c - s) * dd - 0.5f * (float)n - 0.001f);
    c2 = c1 + n - 1;

    v1 = min(max(c1,  0), t    );
    v2 = min(max(c2, -1), t - 1);

    return (v1 <= v2);

}

/*--------------------------------------------------------------------------------------------------------------------*/

INLINE bool _RFNear(int t, float s, float d, float c, int n, int &v1, int &v2) {

    int c1, c2;
    _RFNear(t, s, d, c, n, v1, v2, c1, c2);

    return (v1 == c1) && (v2 == c2);

}
