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

#include "common_dec.h"

class _Session;

#include "util_dec.h"

INLINEH unsigned int _Roll(int c, unsigned int roll, unsigned int size);
INLINE  unsigned int _Unroll(unsigned int c, unsigned int roll, unsigned int size);

INLINE void _DivMod(int &a, unsigned int b, int &d);
INLINE void _ShiftMod(int &a, unsigned int b, int &d);

typedef struct {unsigned short x, y, z, w, a, b, c, d;} ushort8;

/**********************************************************************************************************************/

// Use all available constant memory, currently limited to 64K bytes.  It would be nice to query the device for the
// amount of constant memory available.

const unsigned int _MAX_CDATA = 8192;
const unsigned int _MAX_CMETA = 16384;

// These are fairly arbitrary limits governing the size of statically allocated arrays in host memory.  There would not
// be much penalty for increasing them, and they could be removed altogether if we were willing to make the code a bit
// more complicated (i.e. use dynamic allocation and deallocation).

const unsigned int _MAX_SESSIONS = 10;
const unsigned int _MAX_KERNELS  = 100;
const unsigned int _MAX_RESULTS  = 100;
const unsigned int _NAME_LEN     = 32;

/**********************************************************************************************************************/

class _T_BASE {
public:
    int z;
};

INLINE _T_BASE _P_BASE(int z) {
    _T_BASE p;
    p.z = z;
    return p;
}

INLINE int _Z_BASE(_T_BASE p) {
    return p.z;
}

/**********************************************************************************************************************/

#include _USER_GLOBAL_DEC

/**********************************************************************************************************************/

class _OutTable {
public:

    float        *ptr[_NUM_CV_NZ];
    unsigned int  h  [_NUM_CV_NZ];
    unsigned int  w  [_NUM_CV_NZ];

};

/**********************************************************************************************************************/

// We put this structure in union with an array of uints so that the structure can be read in by multiple threads in
// parallel using a single coalesced read.  The structure must be exactly 256 bytes so that each structure in an array
// of structures will be correctly aligned for coalesced reading.

const unsigned int _LAYERDATA_UINTS = 32;

class _LayerData {
public:

    union {

        struct {

            unsigned int        m_yCount;
            unsigned int        m_ySize;
            unsigned int        m_xCount;
            unsigned int        m_sSize;

            unsigned int        m_gmvOff;
            unsigned int        m_mvOff;
            unsigned int        m_gcOff;
            unsigned int        m_cOff;

            unsigned int        m_tOff;

            float              *m_ndPtr;
            float              *m_nwPtr;
            float              *m_nwOut;
            const unsigned int *m_nmPtr;
            float              *m_sdPtr;
            const ushort4      *m_smPtr;

        };

        unsigned int m_array[_LAYERDATA_UINTS];

    };

};

/**********************************************************************************************************************/

class _Layer : private _LayerData {
public:

    void LInit(_Session *s, const mxArray *layers, unsigned int z);

public:

    #ifndef _GPU
        void LRun(unsigned int kerNo, unsigned int yStart, unsigned int yCount, unsigned int yStep,
            unsigned int xStart, unsigned int xCount, unsigned int xStep);
    #endif

public:

    float *GetNDPtr() { return m_ndPtr; };
    float *GetNWOut() { return m_nwOut; };
    float *GetSDPtr() { return m_sdPtr; };

    bool Shiftable() { return m_shLFlag; };

    unsigned int GetShStartOff() { return m_shStartOff; };
    unsigned int GetShShiftOff() { return m_shShiftOff; };
    unsigned int GetShRollOff () { return m_shRollOff ; };

    unsigned int GetShift() { return m_shift; };
    void SetShift(unsigned int shift) { m_shift = shift; };

    unsigned int GetRoll() { return m_roll; };
    void SetRoll(unsigned int roll) { m_roll = roll; };

private:

    // We rely on these members being at the end of the structure.

    _Session     *m_s;

    unsigned int  m_z;

    bool          m_shLFlag;
    unsigned int  m_shStartOff;
    unsigned int  m_shShiftOff;
    unsigned int  m_shRollOff;

    unsigned int  m_shift;
    unsigned int  m_roll;

};

/**********************************************************************************************************************/

class _Kernel {
public:

    void KInit(_Session *s, const mxArray *kernels, unsigned int k);

public:

    void KRun();

private:

    void KRun2(unsigned int bStart, unsigned int bCount);

public:

    unsigned int GetStepNo() { return m_stepNo; };

private:

    _Session      *m_s;

    char           m_type[_NAME_LEN];
    char           m_ker [_NAME_LEN];

    unsigned int   m_stepNo;
    unsigned int   m_kerNo;

    unsigned int   m_blockSize;
    unsigned int   m_z0;

    unsigned int   m_bCount;
    const ushort8 *m_bPtr;

    bool           m_shKFlag;
    unsigned int   m_bShift;

};

/**********************************************************************************************************************/

class _Result {
public:

    void RInit(_Session *s, const mxArray *p, unsigned int resultNo, unsigned int totalSamples, bool hold);
    void RAlloc(mxArray *&res);
    void RAttach(const mxArray *res);

public:

    void RInit2();
    void RDone();

    void SampleToHold(unsigned int sampleNo);
    void HoldToResult();
    void SampleToResult();
    void Update();

private:

    _Session     *m_s;

    unsigned int  m_varType;

    unsigned int  m_pos;

    unsigned int  m_sBufH;
    unsigned int  m_sBufW;

    unsigned int  m_hOff;
    unsigned int  m_hCount;

    unsigned int  m_wSeg;
    unsigned int  m_wOffs  [2];
    unsigned int  m_wCounts[2];

    unsigned int  m_dOff;
    unsigned int  m_dCount;

    unsigned int  m_totalSamples;
    bool          m_hold;

    float        *m_sBuf;

    unsigned int  m_rBufH;
    unsigned int  m_rBufW;

    unsigned int  m_rCount;

    float        *m_hBuf;

    float        *m_rBuf;

};

/**********************************************************************************************************************/

class _WorkerArgs {
private:
    unsigned int dummy;
};

typedef void (_Session::*_WorkerMethod)(const _WorkerArgs *);

/**********************************************************************************************************************/

#include _SYNC_INC

class _Session {
public:

    // Master thread methods.

    static _Session *New(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    static void Delete(_Session *s);

    void Master(int mode, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

private:

    // Master thread methods.

    _Session();

    bool Claim  (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    bool Release(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    bool Init   (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    bool Done   (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    bool Run    (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    bool Output (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    bool Get    (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    bool Set    (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    bool Shift  (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    bool Wait   (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    void Poll   (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

    void Shift1(const mxArray *sh);

    void Setup(unsigned int exitLevel, bool checkErr);
    void Cleanup();

public:

    // Utility methods called by master thread.

    void Exit(const char *format, ...);

    static void Unlock();

private:

    // Utility methods called by master thread.

    void Error(bool prev);

    static void Lock();
    static void AtExit();

    void Worker(_WorkerMethod method, const _WorkerArgs *args, bool wait);

private:

    // Synchronization methods called by master thread.

    bool CreateWorker(void *(*f)(void *));
    void DestroyWorker();
    void RunWorker();
    bool IsWorkerRunning();
    void WaitForWorker();

    // Synchronization methods called by worker threads.

    void WaitForMaster();
    void ReleaseMaster();
    void Die();

private:

    // Worker thread methods.

    static void *WorkerMain(void *ptr);

    void Worker2();

    void Claim2 (const _WorkerArgs *args);
    void Init2  (const _WorkerArgs *args);
    void Init3  (const _WorkerArgs *args);
    void Run2   (const _WorkerArgs *args);
    void Output2(const _WorkerArgs *args);
    void Get2   (const _WorkerArgs *args);
    void Set2   (const _WorkerArgs *args);
    void Shift2 (const _WorkerArgs *args);

    void Cleanup2(const _WorkerArgs *args);

public:

    // Utility methods called by worker threads.

    void Exit2(const char *format, ...);
    void NeuronExit(unsigned int z, unsigned int y, unsigned int x, const char *format, ...);
    void NeuronInfo(unsigned int z, unsigned int y, unsigned int x, const char *format, ...);

public:

    // Accessor methods.

    const _DeviceProps *GetProps() { return &m_props; };

    unsigned int GetMMVOff() { return m_mmvOff; };
    unsigned int GetMCOff () { return m_mcOff ; };

    void GetCDataBuf(float *ptr, unsigned int off, unsigned int count) {
        memcpy(ptr, m_cDataBuf + off, count * sizeof(float));
    }
    void SetCDataBuf(unsigned int off, const float *ptr, unsigned int count) {
        memcpy(m_cDataBuf + off, ptr, count * sizeof(float));
    }

    #ifndef _GPU
        float          *GetCData() { return _G_CDATASYM; };
        unsigned short *GetCMeta() { return _G_CMETASYM; };
    #endif

    _OutTable &GetTOut() { return m_tOut; };
    float *GetTOutPtr(unsigned int pos) { return m_tOut.ptr[pos]; };

    float              *GetDData    (unsigned int offset) { return m_dData     + offset; };
    float              *GetDWData   (unsigned int offset) { return m_dWData    + offset; };
    float              *GetDWOut    (unsigned int offset) { return m_dWOut     + offset; };
    const unsigned int *GetDNeurons (unsigned int offset) { return m_dNeurons  + offset; };
    const ushort4      *GetDSynapses(unsigned int offset) { return m_dSynapses + offset; };
    const ushort8      *GetDBlocks  (unsigned int offset) { return m_dBlocks   + offset; };

    _LayerData *GetDLayers(unsigned int z) { return m_dLayers + z; };

    _Layer *GetLayer(unsigned int z) { return m_layers[z]; };

    unsigned int GetIterNo() { return m_iterNo; };
    void SetIterNo(unsigned int no) { m_iterNo = no; };

    void RegisterResult(unsigned int resultNo) { m_numResults = resultNo + 1; };

private:

    // Data members.

    static unsigned int  _g_numSessions;
    static _Session     *_g_sessions[_MAX_SESSIONS];

    unsigned int m_exitLevel;
    unsigned int m_initLevel;

    unsigned int m_err;
    char         m_errMsg[_ERRMSG_LEN];
    unsigned int m_errZ;
    unsigned int m_errY;
    unsigned int m_errX;

    bool               m_multi;
    unsigned int       m_kill;
    _WorkerMethod      m_method;
    const _WorkerArgs *m_args;

    bool         m_debug;
    _DeviceProps m_props;

    mxArray        *m_CB;

    bool            m_independent;

    unsigned int    m_mmvOff;
    unsigned int    m_mcOff;

    float           m_cDataBuf[_MAX_CDATA];
    unsigned short  m_cMetaBuf[_MAX_CMETA];
    unsigned int    m_cDataNum;
    unsigned int    m_cMetaNum;

    #ifndef _GPU
        float          _G_CDATASYM[_MAX_CDATA];
        unsigned short _G_CMETASYM[_MAX_CMETA];
    #endif

    _OutTable       m_tOut;

    unsigned int    m_nwCount;

    float          *m_dData;
    float          *m_dWData;
    float          *m_dWOut;
    unsigned int   *m_dNeurons;
    ushort4        *m_dSynapses;
    ushort8        *m_dBlocks;
    _LayerData     *m_dLayers;

    unsigned int    m_lCount;
    _Layer         *m_layers[_MAX_LAYERS];

    unsigned int    m_kCount;
    _Kernel        *m_kernels[_MAX_KERNELS];

    unsigned int    m_iterNo;

    bool            m_shPFlag;
    unsigned int    m_shCurrentOff;

    unsigned int    m_numResults;
    _Result         m_results[_MAX_RESULTS];

    #include _SYNC_DEC

    #include _USER_SESSION

};

unsigned int  _Session::_g_numSessions = 0;
_Session     *_Session::_g_sessions[_MAX_SESSIONS];

/**********************************************************************************************************************/

#include _USER_ALLKERNELS_DEC

/**********************************************************************************************************************/

#ifdef _GPU
    __constant__ float          _G_CDATASYM[_MAX_CDATA];
    __constant__ unsigned short _G_CMETASYM[_MAX_CMETA];
#endif

/***********************************************************************************************************************
* MASTER THREAD FUNCTIONS
***********************************************************************************************************************/

#ifdef _BITS_32
    // Needed to suppress warning on 32-bit systems.
    typedef unsigned int _PtrAsInt;
#else
    typedef unsigned long long _PtrAsInt;
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    if (nrhs == 0) {
        plhs[0] = mxCreateDoubleScalar(_VERSION);
        return;
    }

    int mode = (int)mxGetScalar(prhs[0]);

    _Session *s;

    switch (mode) {
    case -1:
        _Session::Unlock();
        break;
    case 0:
        s = _Session::New(nlhs - 1, plhs + 1, nrhs - 1, prhs + 1);
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        *(unsigned long long *)mxGetData(plhs[0]) = (unsigned long long)s;
        break;
    case 1:
        s = (_Session *)(_PtrAsInt)*(unsigned long long *)mxGetData(prhs[1]);
        _Session::Delete(s);
        break;
    default:
        s = (_Session *)(_PtrAsInt)*(unsigned long long *)mxGetData(prhs[1]);
        s->Master(mode, nlhs, plhs, nrhs - 2, prhs + 2);
        break;
    }

}

/**********************************************************************************************************************/

_Session *_Session::New(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    if (_g_numSessions == 0) Lock();

    _Session *s = new _Session();

    _g_sessions[_g_numSessions++] = s;

    s->m_multi = s->CreateWorker(WorkerMain);

    s->m_kill = 0;
    s->Master(0, nlhs, plhs, nrhs, prhs);

    return s;

}

/**********************************************************************************************************************/

void _Session::Delete(_Session *s) {

    unsigned int pos;
    for (pos = 0; (pos < _g_numSessions) && (_g_sessions[pos] != s); pos++);
    if (pos == _g_numSessions) {
        // This can happen if MATLAB is stopped during a 'done' call.
        return;
    }

    s->m_kill = 2;
    s->Master(1, 0, NULL, 0, NULL);

    s->m_kill = 1;
    s->RunWorker();

    s->DestroyWorker();

    delete s;

    _g_numSessions--;
    if (pos < _g_numSessions) _g_sessions[pos] = _g_sessions[_g_numSessions];

}

/**********************************************************************************************************************/

void _Session::Master(int mode, int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    if (mode == 10) {
        Poll(nlhs, plhs, nrhs, prhs);
        return;
    }

    WaitForWorker();

    bool waited;

    switch (mode) {
    case 0: waited = Claim  (nlhs, plhs, nrhs, prhs); break;
    case 1: waited = Release(nlhs, plhs, nrhs, prhs); break;
    case 2: waited = Init   (nlhs, plhs, nrhs, prhs); break;
    case 3: waited = Done   (nlhs, plhs, nrhs, prhs); break;
    case 4: waited = Run    (nlhs, plhs, nrhs, prhs); break;
    case 5: waited = Output (nlhs, plhs, nrhs, prhs); break;
    case 6: waited = Get    (nlhs, plhs, nrhs, prhs); break;
    case 7: waited = Set    (nlhs, plhs, nrhs, prhs); break;
    case 8: waited = Shift  (nlhs, plhs, nrhs, prhs); break;
    case 9: waited = Wait   (nlhs, plhs, nrhs, prhs); break;
    }

    if (waited) Cleanup();

}

/**********************************************************************************************************************/

_Session::_Session() {

    m_exitLevel = 0;
    m_initLevel = 0;

    m_err = 0;

}

/**********************************************************************************************************************/

class _ClaimArgs : public _WorkerArgs {
public:

    int  desiredDeviceNo;
    bool nice;

};

/*--------------------------------------------------------------------------------------------------------------------*/

bool _Session::Claim(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    Setup(0, true);

    m_initLevel = 1;

    _ClaimArgs w;

    w.desiredDeviceNo = (int)mxGetScalar(prhs[0]);
    w.nice = ((int)mxGetScalar(prhs[1]) > 0);

    m_debug = ((int)mxGetScalar(prhs[2]) > 0);

    Worker(&_Session::Claim2, &w, true);

    mxArray *p = mxCreateStructMatrix(1, 1, 0, NULL);

    mxSetFieldByNumber(p, 0, mxAddField(p, "deviceNo"       ), mxCreateDoubleScalar(m_props.deviceNo       ));
    mxSetFieldByNumber(p, 0, mxAddField(p, "blockSizeAlign" ), mxCreateDoubleScalar(m_props.blockSizeAlign ));
    mxSetFieldByNumber(p, 0, mxAddField(p, "blockYSizeAlign"), mxCreateDoubleScalar(m_props.blockYSizeAlign));
    mxSetFieldByNumber(p, 0, mxAddField(p, "maxTexYSize"    ), mxCreateDoubleScalar(m_props.maxTexYSize    ));
    mxSetFieldByNumber(p, 0, mxAddField(p, "maxTexXSize"    ), mxCreateDoubleScalar(m_props.maxTexXSize    ));
    mxSetFieldByNumber(p, 0, mxAddField(p, "maxCData"       ), mxCreateDoubleScalar(_MAX_CDATA             ));
    mxSetFieldByNumber(p, 0, mxAddField(p, "maxCMeta"       ), mxCreateDoubleScalar(_MAX_CMETA             ));
    mxSetFieldByNumber(p, 0, mxAddField(p, "minBlockSize"   ), mxCreateDoubleScalar(_LAYERDATA_UINTS       ));

    plhs[0] = p;

    m_exitLevel = 1;

    return true;

}

/**********************************************************************************************************************/

bool _Session::Release(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    Setup(0, false);

    return true;

}

/**********************************************************************************************************************/

class _InitArgs : public _WorkerArgs {
public:

    const float          *tTDataPtr[_NUM_T_NZ];
    unsigned int          tTDataM  [_NUM_T_NZ];
    unsigned int          tTDataN  [_NUM_T_NZ];

    const float          *tCDataPtr[_NUM_CC_NZ];
    unsigned int          tCDataM  [_NUM_CC_NZ];
    unsigned int          tCDataN  [_NUM_CC_NZ];

    const float          *tVDataPtr[_NUM_CV_NZ];

    const float          *dDataPtr;
    const float          *dWDataPtr;
    const unsigned int   *dNeuronsPtr;
    const ushort4        *dSynapsesPtr;
    const ushort8        *dBlocksPtr;
    unsigned int          dDataNum;
    unsigned int          dWDataNum;
    unsigned int          dNeuronsNum;
    unsigned int          dSynapsesNum;
    unsigned int          dBlocksNum;

};

/*--------------------------------------------------------------------------------------------------------------------*/

bool _Session::Init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    Setup(1, true);

    m_initLevel = 2;

    m_CB = NULL;

    for (int f = 0; f < _NUM_T; f++) {
        _ClearTex(*GetTTexA(f), *GetTTexB(f));
    }

    for (int f = 0; f < _NUM_CC; f++) {
        _ClearTex(*GetCTexA(f), *GetCTexB(f));
    }

    for (int f = 0; f < _NUM_CV; f++) {
        _ClearTex(*GetVTexA(f), *GetVTexB(f));
        _ClearBuf(m_tOut.ptr[f]);
    }

    _ClearBuf(m_dData    );
    _ClearBuf(m_dWData   );
    _ClearBuf(m_dWOut    );
    _ClearBuf(m_dNeurons );
    _ClearBuf(m_dSynapses);
    _ClearBuf(m_dBlocks  );
    _ClearBuf(m_dLayers  );

    m_lCount = 0;
    m_kCount = 0;

    _InitArgs w;

    // Note this copy won't be updated after initialization.
    m_CB = mxDuplicateArray(prhs[0]);
    mexMakeArrayPersistent(m_CB);

    const mxArray *s = prhs[1];
    const mxArray *h = prhs[2];

    m_independent = ((int)mxGetScalar(mxGetField(s, 0, "independent")) > 0);

    m_mmvOff = (unsigned int)mxGetScalar(mxGetField(s, 0, "mvOff"));
    m_mcOff  = (unsigned int)mxGetScalar(mxGetField(s, 0, "cOff" ));

    const mxArray *cData = mxGetField(h, 0, "cData");
    const mxArray *cMeta = mxGetField(h, 0, "cMeta");
    m_cDataNum = mxGetNumberOfElements(cData);
    m_cMetaNum = mxGetNumberOfElements(cMeta);
    memcpy(m_cDataBuf, mxGetData(cData), m_cDataNum * sizeof(float         ));
    memcpy(m_cMetaBuf, mxGetData(cMeta), m_cMetaNum * sizeof(unsigned short));

    const mxArray *tTData = mxGetField(h, 0, "tTData");
    for (int f = 0; f < _NUM_T; f++) {
        const mxArray *tex = mxGetCell(tTData, f);
        w.tTDataPtr[f] = (const float *)mxGetData(tex);
        w.tTDataM  [f] = mxGetM(tex);
        w.tTDataN  [f] = mxGetN(tex);
    }

    const mxArray *tCData = mxGetField(h, 0, "tCData");
    for (int f = 0; f < _NUM_CC; f++) {
        const mxArray *tex = mxGetCell(tCData, f);
        w.tCDataPtr[f] = (const float *)mxGetData(tex);
        w.tCDataM  [f] = mxGetM(tex);
        w.tCDataN  [f] = mxGetN(tex);
    }

    const mxArray *tVData = mxGetField(h, 0, "tVData");
    for (int f = 0; f < _NUM_CV; f++) {
        const mxArray *tex = mxGetCell(tVData, f);
        w.tVDataPtr[f] = (const float *)mxGetData(tex);
        m_tOut.h   [f] = mxGetM(tex);
        m_tOut.w   [f] = mxGetN(tex);
    }

    m_nwCount = mxGetNumberOfElements(mxGetField(h, 0, "dWData"));

    w.dDataPtr     = (const float *       )mxGetData(mxGetField(h, 0, "dData"    ));
    w.dWDataPtr    = (const float *       )mxGetData(mxGetField(h, 0, "dWData"   ));
    w.dNeuronsPtr  = (const unsigned int *)mxGetData(mxGetField(h, 0, "dNeurons" ));
    w.dSynapsesPtr = (const ushort4      *)mxGetData(mxGetField(h, 0, "dSynapses"));
    w.dBlocksPtr   = (const ushort8      *)mxGetData(mxGetField(h, 0, "dBlocks"  ));
    w.dDataNum     = mxGetNumberOfElements(mxGetField(h, 0, "dData"    ));
    w.dWDataNum    = mxGetNumberOfElements(mxGetField(h, 0, "dWData"   ));
    w.dNeuronsNum  = mxGetNumberOfElements(mxGetField(h, 0, "dNeurons" ));
    w.dSynapsesNum = mxGetNumberOfElements(mxGetField(h, 0, "dSynapses")) / 4;
    w.dBlocksNum   = mxGetNumberOfElements(mxGetField(h, 0, "dBlocks"  )) / 8;

    Worker(&_Session::Init2, &w, true);

    const mxArray *layers = mxGetField(s, 0, "layers");
    unsigned int lCount = mxGetNumberOfElements(layers);
    if (lCount > _MAX_LAYERS) {
        Exit("maximum number of layers (%u) exceeded", _MAX_LAYERS);
    }
    for (unsigned int z = 0; z < lCount; z++) {
        m_layers[z] = new _Layer();
        m_lCount = z + 1;
        m_layers[z]->LInit(this, layers, z);
    }

    if (sizeof(_LayerData) != _LAYERDATA_UINTS * sizeof(unsigned int)) {
        Exit("size of LayerData structure is not %u bytes", _LAYERDATA_UINTS * sizeof(unsigned int));
    }

    Worker(&_Session::Init3, NULL, true);

    const mxArray *kernels = mxGetField(s, 0, "kernels");
    unsigned int kCount = mxGetNumberOfElements(kernels);
    if (kCount > _MAX_KERNELS) {
        Exit("maximum number of kernels (%u) exceeded", _MAX_KERNELS);
    }
    for (unsigned int k = 0; k < kCount; k++) {
        m_kernels[k] = new _Kernel();
        m_kCount = k + 1;
        m_kernels[k]->KInit(this, kernels, k);
    }

    m_iterNo = 0;

    m_shPFlag = (mxGetNumberOfElements(mxGetField(s, 0, "sn")) != 0);

    m_shCurrentOff = (unsigned int)mxGetScalar(mxGetField(s, 0, "shCurrentOff"));

    if (m_shPFlag) {
        Shift1(mxGetField(s, 0, "sh"));
    }

    m_exitLevel = 2;

    return true;

}

/**********************************************************************************************************************/

bool _Session::Done(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    Setup(1, false);

    return true;

}

/**********************************************************************************************************************/

class _RunArgs : public _WorkerArgs {
public:

    unsigned int iters;
    unsigned int sampleRate;

    unsigned int step1;
    unsigned int step2;
    bool         increment;

};

/*--------------------------------------------------------------------------------------------------------------------*/

bool _Session::Run(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    Setup(2, true);

    m_initLevel = 3;

    m_numResults = 0;

    _RunArgs w;

    w.iters      = (unsigned int)mxGetScalar(prhs[0]);
    w.sampleRate = (unsigned int)mxGetScalar(prhs[1]);

    if ((unsigned int)mxGetScalar(prhs[2]) == 0) {
        w.step1     = 0;
        w.step2     = m_kCount; // Essentially, infinity.  Will never be less than the number of steps.
        w.increment = true;
    } else {
        w.step1     = (unsigned int)mxGetScalar(prhs[2]) - 1;
        w.step2     = (unsigned int)mxGetScalar(prhs[3]) - 1;
        w.increment = false;
    }

    unsigned int totalSamples = w.iters / w.sampleRate;

    unsigned int numResults = mxGetNumberOfElements(prhs[4]);
    if (numResults > _MAX_RESULTS) {
        Exit("maximum number of results (%u) exceeded", _MAX_RESULTS);
    }

    for (unsigned int i = 0; i < numResults; i++) {
        m_results[i].RInit(this, prhs[4], i, totalSamples, true);
    }

    Worker(&_Session::Run2, &w, m_debug);

    return m_debug;

}

/**********************************************************************************************************************/

bool _Session::Output(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    if (m_err > 0) Error(true);

    if (m_initLevel < 3) Exit("no output available");

    if (nlhs != m_numResults) Exit("%u outputs needed", m_numResults);

    for (unsigned int i = 0; i < m_numResults; i++) {
        m_results[i].RAlloc(plhs[i]);
    }

    Worker(&_Session::Output2, NULL, true);

    m_exitLevel = 2;

    return true;

}

/**********************************************************************************************************************/

bool _Session::Get(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    Setup(2, true);

    m_initLevel = 3;

    m_numResults = 0;

    m_results[0].RInit(this, prhs[0], 0, 1, false);
    m_results[0].RAlloc(plhs[0]);

    Worker(&_Session::Get2, NULL, true);

    return true;

}

/**********************************************************************************************************************/

bool _Session::Set(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    Setup(2, true);

    m_initLevel = 3;

    m_numResults = 0;

    m_results[0].RInit(this, prhs[0], 0, 1, false);
    m_results[0].RAttach(prhs[1]);

    Worker(&_Session::Set2, NULL, true);

    return true;

}

/**********************************************************************************************************************/

bool _Session::Shift(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    Setup(2, true);

    Shift1(prhs[0]);

    return true;

}

/*--------------------------------------------------------------------------------------------------------------------*/

void _Session::Shift1(const mxArray *sh) {

    m_cDataBuf[m_shCurrentOff] = (float)mxGetScalar(mxGetField(sh, 0, "current"));

    const double *startPtr = mxGetPr(mxGetField(sh, 0, "start"));
    const double *shiftPtr = mxGetPr(mxGetField(sh, 0, "shift"));
    const double *rollPtr  = mxGetPr(mxGetField(sh, 0, "roll" ));

    for (unsigned int z = 0; z < m_lCount; z++) {
        _Layer *lay = m_layers[z];
        if (!lay->Shiftable()) continue;
        m_cDataBuf[lay->GetShStartOff()] = (float)startPtr[z];
        m_cDataBuf[lay->GetShShiftOff()] = _IntAsFloat((int)shiftPtr[z]);
        m_cMetaBuf[lay->GetShRollOff ()] = (unsigned short)rollPtr[z];
        lay->SetShift((unsigned int)shiftPtr[z]);
        lay->SetRoll ((unsigned int)rollPtr [z]);
    }

    Worker(&_Session::Shift2, NULL, true);

}

/**********************************************************************************************************************/

bool _Session::Wait(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // Nothing to do.

    return true;

}

/**********************************************************************************************************************/

void _Session::Poll(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    plhs[0] = mxCreateDoubleScalar(IsWorkerRunning() ? 1.0 : 0.0);

}

/**********************************************************************************************************************/

void _Session::Setup(unsigned int exitLevel, bool checkErr) {

    if (checkErr && (m_err > 0)) Error(true);

    m_exitLevel = exitLevel;

    m_err = 0;

    Cleanup();

}

/**********************************************************************************************************************/

void _Session::Cleanup() {

    if (m_initLevel <= m_exitLevel) return;

    Worker(&_Session::Cleanup2, NULL, true);

    if ((m_exitLevel < 2) && (m_initLevel >= 2)) {

        if (m_CB != NULL) {
            mxDestroyArray(m_CB);
            m_CB = NULL;
        }

        for (unsigned int z = 0; z < m_lCount; z++) {
            delete m_layers[z];
        }
        m_lCount = 0;

        for (unsigned int k = 0; k < m_kCount; k++) {
            delete m_kernels[k];
        }
        m_kCount = 0;

    }

    m_initLevel = m_exitLevel;

}

/**********************************************************************************************************************/

void _Session::Exit(const char *format, ...) {

    m_err = 1;

    va_list argList;
    va_start(argList, format);
    vsprintf(m_errMsg, format, argList);
    va_end(argList);

    Error(false);

}

/**********************************************************************************************************************/

void _Session::Error(bool prev) {

    char msg[_ERRMSG_LEN];
    char buf[_ERRMSG_LEN];

    if (prev) {
        strcpy(msg, "(previous call) ");
    } else {
        strcpy(msg, "");
    }

    switch (m_err) {
    case 1:
        sprintf(msg + strlen(msg), "%s", m_errMsg);
        break;
    case 2:
        sprintf(msg + strlen(msg), "ITER_NO=%u Z=%u %s: %s",
            m_iterNo, m_errZ,
            _CBYX2S(m_CB, m_errZ, m_errY, m_errX, m_layers[m_errZ]->GetRoll(), true, buf),
            m_errMsg);
        break;
    }

    m_err = 0;

    Cleanup();

    mexErrMsgTxt(msg);

}

/**********************************************************************************************************************/

void _Session::Lock() {

    if (!mexIsLocked()) mexLock();

    mexAtExit(AtExit);

}

/**********************************************************************************************************************/

void _Session::Unlock() {

    if (mexIsLocked()) mexUnlock();

    AtExit();

}

/**********************************************************************************************************************/

void _Session::AtExit() {

    while (_g_numSessions > 0) {
        Delete(_g_sessions[--_g_numSessions]);
    }

}

/**********************************************************************************************************************/

void _Session::Worker(_WorkerMethod method, const _WorkerArgs *args, bool wait) {

    bool multi = m_multi;

    #ifndef _GPU
    if (wait) multi = false;
    #endif

    if (multi) {

        m_method = method;
        m_args   = args;

        RunWorker();

    } else {

        try {
            (this->*method)(args);
        } catch (...) {
        }

    }

    if (wait && (m_err > 0)) Error(false);

}

/**********************************************************************************************************************/

void _Layer::LInit(_Session *s, const mxArray *layers, unsigned int z) {

    m_s = s;

    m_yCount = (unsigned int)mxGetScalar(mxGetField(layers, z, "yCount"    ));
    m_ySize  = (unsigned int)mxGetScalar(mxGetField(layers, z, "ySize"     ));
    m_xCount = (unsigned int)mxGetScalar(mxGetField(layers, z, "xCount"    ));
    m_sSize  = (unsigned int)mxGetScalar(mxGetField(layers, z, "sSize"     ));
    m_gmvOff = (unsigned int)mxGetScalar(mxGetField(layers, z, "gmvOff"    ));
    m_mvOff  = (unsigned int)mxGetScalar(mxGetField(layers, z, "mvOff"     ));
    m_gcOff  = (unsigned int)mxGetScalar(mxGetField(layers, z, "gcOff"     ));
    m_cOff   = (unsigned int)mxGetScalar(mxGetField(layers, z, "cOff"      ));
    m_tOff   = (unsigned int)mxGetScalar(mxGetField(layers, z, "tOff"      ));

    m_ndPtr = m_s->GetDData    ((unsigned int)mxGetScalar(mxGetField(layers, z, "ndOff")));
    m_nwPtr = m_s->GetDWData   ((unsigned int)mxGetScalar(mxGetField(layers, z, "nwOff")));
    m_nwOut = m_s->GetDWOut    ((unsigned int)mxGetScalar(mxGetField(layers, z, "nwOff")));
    m_nmPtr = m_s->GetDNeurons ((unsigned int)mxGetScalar(mxGetField(layers, z, "nmOff")));
    m_sdPtr = m_s->GetDData    ((unsigned int)mxGetScalar(mxGetField(layers, z, "sdOff")));
    m_smPtr = m_s->GetDSynapses((unsigned int)mxGetScalar(mxGetField(layers, z, "smOff")));

    m_z          = z;
    m_shStartOff = (unsigned int)mxGetScalar(mxGetField(layers, z, "shStartOff"));
    m_shShiftOff = (unsigned int)mxGetScalar(mxGetField(layers, z, "shShiftOff"));
    m_shRollOff  = (unsigned int)mxGetScalar(mxGetField(layers, z, "shRollOff" ));

    m_shLFlag = ((unsigned int)mxGetScalar(mxGetField(layers, z, "xShift")) != 0);

    m_shift = 0;
    m_roll  = 0;

}

/**********************************************************************************************************************/

void _Kernel::KInit(_Session *s, const mxArray *kernels, unsigned int k) {

    m_s = s;

    mxGetString(mxGetField(kernels, k, "type"), m_type, _NAME_LEN);
    mxGetString(mxGetField(kernels, k, "ker" ), m_ker , _NAME_LEN);

    m_stepNo    = (unsigned int)mxGetScalar(mxGetField(kernels, k, "stepNo"   )) - 1;
    m_kerNo     = (unsigned int)mxGetScalar(mxGetField(kernels, k, "kerNo"    )) - 1;
    m_blockSize = (unsigned int)mxGetScalar(mxGetField(kernels, k, "blockSize"));
    m_z0        = (unsigned int)mxGetScalar(mxGetField(kernels, k, "z0"       )) - 1;
    m_bCount    = (unsigned int)mxGetScalar(mxGetField(kernels, k, "bCount"   ));
    m_bShift    = (unsigned int)mxGetScalar(mxGetField(kernels, k, "bShift"   ));

    m_bPtr = m_s->GetDBlocks((unsigned int)mxGetScalar(mxGetField(kernels, k, "bOff")));

    m_shKFlag = (m_bShift != 0);

}

/**********************************************************************************************************************/

void _Result::RInit(_Session *s, const mxArray *p, unsigned int resultNo, unsigned int totalSamples, bool hold) {

    m_s = s;

    m_varType = (unsigned int)mxGetScalar(mxGetField(p, resultNo, "varType"));
    m_pos     = (unsigned int)mxGetScalar(mxGetField(p, resultNo, "pos"    ));
    m_sBufH   = (unsigned int)mxGetScalar(mxGetField(p, resultNo, "height" ));
    m_sBufW   = (unsigned int)mxGetScalar(mxGetField(p, resultNo, "width"  ));
    m_hOff    = (unsigned int)mxGetScalar(mxGetField(p, resultNo, "hOff"   ));
    m_hCount  = (unsigned int)mxGetScalar(mxGetField(p, resultNo, "hCount" ));
    m_dOff    = (unsigned int)mxGetScalar(mxGetField(p, resultNo, "dOff"   ));
    m_dCount  = (unsigned int)mxGetScalar(mxGetField(p, resultNo, "dCount" ));

    m_wSeg = mxGetNumberOfElements(mxGetField(p, resultNo, "wOff"));
    double *wOffs   = mxGetPr(mxGetField(p, resultNo, "wOff"  ));
    double *wCounts = mxGetPr(mxGetField(p, resultNo, "wCount"));
    for (unsigned int i = 0; i < m_wSeg; i++) {
        m_wOffs  [i] = (unsigned int)wOffs  [i];
        m_wCounts[i] = (unsigned int)wCounts[i];
    }

    m_totalSamples = totalSamples;
    m_hold         = hold;

    switch (m_varType) {
    case 0: m_sBuf = NULL                            ; break; // *p
    case 1: m_sBuf = m_s->GetDData(m_pos)            ; break; // *a
    case 2: m_sBuf = NULL                            ; break; // *t
    case 3: m_sBuf = NULL                            ; break; // cc
    case 4: m_sBuf = m_s->GetTOutPtr(m_pos)          ; break; // cv
    case 5: m_sBuf = m_s->GetLayer(m_pos)->GetNDPtr(); break; // nc, nv
    case 6: m_sBuf = m_s->GetLayer(m_pos)->GetNWOut(); break; // nw
    case 7: m_sBuf = m_s->GetLayer(m_pos)->GetSDPtr(); break; // s*
    case 8: m_sBuf = NULL                            ; break; // sp
    }

    double *siz4b = mxGetPr(mxGetField(mxGetField(p, resultNo, "t"), 0, "siz4b"));
    m_rBufH = (unsigned int)siz4b[0];
    m_rBufW = (unsigned int)siz4b[1];

    m_rCount = m_rBufH * m_rBufW * m_dCount;

    _ClearBuf(m_hBuf);

    m_s->RegisterResult(resultNo);

}

/**********************************************************************************************************************/

void _Result::RAlloc(mxArray *&res) {

    res = mxCreateNumericMatrix(m_rCount, m_totalSamples, mxSINGLE_CLASS, mxREAL);

    m_rBuf = (float *)mxGetData(res);

}

/**********************************************************************************************************************/

void _Result::RAttach(const mxArray *res) {

    m_rBuf = (float *)mxGetData(res);

}

/***********************************************************************************************************************
* WORKER THREAD FUNCTIONS
***********************************************************************************************************************/

void *_Session::WorkerMain(void *ptr) {

    _Session *s = (_Session *)ptr;

    for (;;) {

        s->WaitForMaster();

        if (s->m_kill == 1) break;

        s->Worker2();

    }

    s->Die();

    return NULL;

}

/**********************************************************************************************************************/

void _Session::Worker2() {

    try {
        (this->*m_method)(m_args);
    } catch (...) {
    }

}

/**********************************************************************************************************************/

void _Session::Claim2(const _WorkerArgs *args) {

    const _ClaimArgs &w = *(const _ClaimArgs *)args;

    _ClaimDevice(this, w.desiredDeviceNo, w.nice, m_props);

}

/**********************************************************************************************************************/

void _Session::Init2(const _WorkerArgs *args) {

    const _InitArgs &w = *(const _InitArgs *)args;

    _AllocConst(this, _G_CDATASTR, _G_CDATASYM, m_cDataBuf, m_cDataNum);
    _AllocConst(this, _G_CMETASTR, _G_CMETASYM, m_cMetaBuf, m_cMetaNum);

    for (int f = 0; f < _NUM_T; f++) {
        _AllocTexArray(this, *GetTTexA(f), *GetTTexB(f),
            w.tTDataPtr[f], w.tTDataM[f], w.tTDataN[f], "texture constants");
    }

    for (int f = 0; f < _NUM_CC; f++) {
        _AllocTexArray(this, *GetCTexA(f), *GetCTexB(f),
            w.tCDataPtr[f], w.tCDataM[f], w.tCDataN[f], "common constants");
    }

    for (int f = 0; f < _NUM_CV; f++) {
        _AllocBuf(this, m_tOut.ptr[f], w.tVDataPtr[f], m_tOut.h[f] * m_tOut.w[f], "common variables output");
        if (m_independent) {
            _AllocTexLinear(this, *GetVTexA(f), *GetVTexB(f),
                m_tOut.ptr[f], m_tOut.h[f], m_tOut.w[f], "common variables");
        } else {
            _AllocTexArray (this, *GetVTexA(f), *GetVTexB(f),
                NULL         , m_tOut.h[f], m_tOut.w[f], "common variables");
        }
    }

    _AllocBuf(this, m_dData    , w.dDataPtr    , w.dDataNum    , "device memory");
    _AllocBuf(this, m_dWOut    , w.dWDataPtr   , w.dWDataNum   , "cell variables output");
    _AllocBuf(this, m_dNeurons , w.dNeuronsPtr , w.dNeuronsNum , "neurons");
    _AllocBuf(this, m_dSynapses, w.dSynapsesPtr, w.dSynapsesNum, "synapses");
    _AllocBuf(this, m_dBlocks  , w.dBlocksPtr  , w.dBlocksNum  , "blocks");

    if (m_independent) {
        m_dWData = m_dWOut;
    } else {
        _AllocBuf(this, m_dWData, (float *)NULL, m_nwCount, "cell variables");
    }

}

/**********************************************************************************************************************/

void _Session::Init3(const _WorkerArgs *args) {

    _LayerData layerData[_MAX_LAYERS];

    for (unsigned int z = 0; z < m_lCount; z++) {
        layerData[z] = *(_LayerData *)m_layers[z];
    }

    _AllocBuf(this, m_dLayers, layerData, m_lCount, "layer data");

}

/**********************************************************************************************************************/

void _Session::Run2(const _WorkerArgs *args) {

    const _RunArgs w = *(const _RunArgs *)args; // we make a copy here

    if (!m_debug) ReleaseMaster();

    for (unsigned int i = 0; i < m_numResults; i++) {
        m_results[i].RInit2();
    }

    unsigned int itersUntilSample = w.sampleRate;
    unsigned int sampleNo         = 0;

    for (unsigned int i = 0; i < w.iters; i++) {

        if (m_kill == 2) return;

        unsigned int prevStep = m_kCount; // Infinity.

        for (unsigned int k = 0; k < m_kCount; k++) {

            unsigned int thisStep = m_kernels[k]->GetStepNo();

            if ((w.step1 <= thisStep) && (thisStep <= w.step2)) {

                if (!m_independent && (prevStep != thisStep)) {
                    for (int f = 0; f < _NUM_CV; f++) {
                        _PublishTex(this, *GetVTexA(f), *GetVTexB(f), m_tOut.ptr[f], m_tOut.h[f], m_tOut.w[f],
                            "common variables");
                    }
                    _CopyBuf1D(this, m_dWData, 'd', m_dWOut, m_nwCount, "cell variables output");
                }

                m_kernels[k]->KRun();

            }

            prevStep = thisStep;

        }

        if (--itersUntilSample == 0) {
            for (unsigned int j = 0; j < m_numResults; j++) {
                m_results[j].SampleToHold(sampleNo);
            }
            itersUntilSample = w.sampleRate;
            sampleNo++;
        }

        if (w.increment) {
            if (m_iterNo < (unsigned int)CNS_INTMAX - 1) {
                m_iterNo++;
            } else {
                m_iterNo = 0;
            }
        }

    }

    m_exitLevel = 3;

}

/**********************************************************************************************************************/

void _Session::Output2(const _WorkerArgs *args) {

    for (unsigned int i = 0; i < m_numResults; i++) {
        m_results[i].HoldToResult();
    }

}

/**********************************************************************************************************************/

void _Session::Get2(const _WorkerArgs *args) {

    m_results[0].SampleToResult();

}

/**********************************************************************************************************************/

void _Session::Set2(const _WorkerArgs *args) {

    m_results[0].Update();

}

/**********************************************************************************************************************/

void _Session::Shift2(const _WorkerArgs *args) {

    _UpdateConst(this, _G_CDATASTR, _G_CDATASYM, 0, m_cDataBuf, m_cDataNum);
    _UpdateConst(this, _G_CMETASTR, _G_CMETASYM, 0, m_cMetaBuf, m_cMetaNum);

}

/**********************************************************************************************************************/

void _Session::Cleanup2(const _WorkerArgs *args) {

    if ((m_exitLevel < 3) && (m_initLevel >= 3)) {

        for (unsigned int i = 0; i < m_numResults; i++) {
            m_results[i].RDone();
        }
        m_numResults = 0;

    }

    if ((m_exitLevel < 2) && (m_initLevel >= 2)) {

        for (int f = 0; f < _NUM_T; f++) {
            _DeallocTex(*GetTTexA(f), *GetTTexB(f));
        }

        for (int f = 0; f < _NUM_CC; f++) {
            _DeallocTex(*GetCTexA(f), *GetCTexB(f));
        }

        for (int f = 0; f < _NUM_CV; f++) {
            _DeallocTex(*GetVTexA(f), *GetVTexB(f));
            _DeallocBuf(m_tOut.ptr[f]);
        }

        _DeallocBuf(m_dData    );
        _DeallocBuf(m_dWOut    );
        _DeallocBuf(m_dNeurons );
        _DeallocBuf(m_dSynapses);
        _DeallocBuf(m_dBlocks  );
        _DeallocBuf(m_dLayers  );

        if (m_independent) {
            _ClearBuf(m_dWData);
        } else {
            _DeallocBuf(m_dWData);
        }

    }

    if ((m_exitLevel < 1) && (m_initLevel >= 1)) {

        _ReleaseDevice();

    }

}

/**********************************************************************************************************************/

void _Session::Exit2(const char *format, ...) {

    m_err = 1;

    va_list argList;
    va_start(argList, format);
    vsprintf(m_errMsg, format, argList);
    va_end(argList);

    throw 1;

}

/**********************************************************************************************************************/

void _Session::NeuronExit(unsigned int z, unsigned int y, unsigned int x, const char *format, ...) {

    m_err = 2;

    va_list argList;
    va_start(argList, format);
    vsprintf(m_errMsg, format, argList);
    va_end(argList);

    m_errZ = z;
    m_errY = y;
    m_errX = x;

    throw 1;

}

/**********************************************************************************************************************/

void _Session::NeuronInfo(unsigned int z, unsigned int y, unsigned int x, const char *format, ...) {

    if (!m_debug) return;

    char pre[_ERRMSG_LEN];
    char buf[_ERRMSG_LEN];

    sprintf(pre, "ITER_NO=%u Z=%u %s",
        m_iterNo, z,
        _CBYX2S(m_CB, z, y, x, m_layers[z]->GetRoll(), true, buf));

    va_list argList;
    va_start(argList, format);
    vsprintf(buf, format, argList);
    va_end(argList);

    mexPrintf("%s: %s\n", pre, buf);

}

/**********************************************************************************************************************/

#ifndef _GPU

    void _Layer::LRun(unsigned int kerNo, unsigned int yStart, unsigned int yCount, unsigned int yStep,
        unsigned int xStart, unsigned int xCount, unsigned int xStep) {

        switch (kerNo) {
        #include _USER_ALLKERNELS_RUN
        default:
            m_s->Exit2("invalid kernel number (%u)", kerNo + 1);
            break;
        }

    }

#endif

/**********************************************************************************************************************/

void _Kernel::KRun() {

    if (m_bCount == 0) return;

    unsigned int bStart, bCount;
    if (m_shKFlag) {
        // Find which blocks to compute based on the shift size & roll value.
        unsigned int size  = m_bCount / m_bShift;
        unsigned int shift = m_s->GetLayer(m_z0)->GetShift();
        unsigned int roll  = m_s->GetLayer(m_z0)->GetRoll ();
        if (shift == 0) return;
        if (shift < size) {
            bStart = m_bShift * _Roll(size - shift, roll, size);
            bCount = m_bShift * shift;
        } else {
            bStart = 0;
            bCount = m_bCount;
        }
    } else {
        // No shiftable dimension.  Compute all blocks.
        bStart = 0;
        bCount = m_bCount;
    }

    KRun2(bStart, bCount);

}

/**********************************************************************************************************************/

#ifdef _GPU

    void _Kernel::KRun2(unsigned int bStart, unsigned int bCount) {

        // The particular 2D shape of the grid and blocks is irrelevant.
        // We use both dimensions to avoid exceeding limits.
        dim3 gridDim, blockDim;
        if (bCount <= 65535) {
            gridDim.y = 1;
            gridDim.x = bCount;
        } else {
            gridDim.y = (unsigned int)sqrt((double)bCount);
            gridDim.x = (unsigned int)ceil((double)bCount / (double)gridDim.y);
        }
        gridDim.z = 1;
        blockDim.y = m_s->GetProps()->blockSizeAlign;
        blockDim.x = m_blockSize / blockDim.y;
        blockDim.z = 1;

        cudaError_t err;

        switch (m_kerNo) {
        #include _USER_ALLKERNELS_RUN
        default:
            m_s->Exit2("invalid kernel number (%u)", m_kerNo + 1);
            break;
        }

    }

#else

    void _Kernel::KRun2(unsigned int bStart, unsigned int bCount) {

        for (unsigned int i = 0, bid = bStart; i < bCount; i++, bid++) {

            if (bid == m_bCount) bid = 0;

            ushort8 b = m_bPtr[bid];

            unsigned int xStart = b.x;
            unsigned int yStart = b.y;
            unsigned int z      = b.z;
            unsigned int xCount = b.a;
            unsigned int yCount = b.b;
            unsigned int xStep  = b.c;
            unsigned int yStep  = b.d;

            m_s->GetLayer(z)->LRun(m_kerNo, yStart, yCount, yStep, xStart, xCount, xStep);

        }

    }

#endif

/**********************************************************************************************************************/

void _Result::RInit2() {

    if (m_hold) {
        _AllocBuf(m_s, m_hBuf, (float *)NULL, m_rCount * m_totalSamples, "hold");
    }

}

/**********************************************************************************************************************/

void _Result::RDone() {

    if (m_hold) {
        _DeallocBuf(m_hBuf);
    }

}

/**********************************************************************************************************************/

void _Result::SampleToHold(unsigned int sampleNo) {

    float *buf = m_hBuf + sampleNo * m_rCount;

    for (unsigned int i = 0, off = 0; i < m_wSeg; off += m_wCounts[i++]) {
        _CopyBuf3D(m_s,
            buf + off * m_rBufH, m_rBufH, m_rBufW, 'd',
            m_sBuf, m_sBufH, m_sBufW, m_hOff, m_wOffs[i], m_dOff,
            m_hCount, m_wCounts[i], m_dCount,
            "variables");
    }

}

/**********************************************************************************************************************/

void _Result::HoldToResult() {

    _CopyBuf1D(m_s, m_rBuf, 'h', m_hBuf, m_rCount * m_totalSamples, "hold");

}

/**********************************************************************************************************************/

void _Result::SampleToResult() {

    switch (m_varType) {
    case 0:
        m_s->GetCDataBuf(m_rBuf, m_hOff, m_hCount);
        break;
    case 1:
    case 4:
    case 5:
    case 6:
    case 7:
        for (unsigned int i = 0, off = 0; i < m_wSeg; off += m_wCounts[i++]) {
            _CopyBuf3D(m_s,
                m_rBuf + off * m_rBufH, m_rBufH, m_rBufW, 'h',
                m_sBuf, m_sBufH, m_sBufW, m_hOff, m_wOffs[i], m_dOff,
                m_hCount, m_wCounts[i], m_dCount,
                "variables");
        }
        break;
    case 2:
        _CopyTex(m_s,
            m_rBuf, m_rBufH, 'h',
            *m_s->GetTTexA(m_pos), *m_s->GetTTexB(m_pos), m_hOff, m_wOffs[0],
            m_hCount, m_wCounts[0],
            "texture constants");
        break;
    case 3:
        _CopyTex(m_s,
            m_rBuf, m_rBufH, 'h',
            *m_s->GetCTexA(m_pos), *m_s->GetCTexB(m_pos), m_hOff, m_wOffs[0],
            m_hCount, m_wCounts[0],
            "common constants");
        break;
    case 8:
        *m_rBuf = _IntAsFloat((int)m_s->GetIterNo());
        break;
    }

}

/**********************************************************************************************************************/

void _Result::Update() {

    switch (m_varType) {
    case 0:
        m_s->SetCDataBuf(m_hOff, m_rBuf, m_hCount);
        #ifdef _GPU
            _UpdateConst(m_s, _G_CDATASTR, _G_CDATASYM    , m_hOff, m_rBuf, m_hCount);
        #else
            _UpdateConst(m_s, _G_CDATASTR, m_s->GetCData(), m_hOff, m_rBuf, m_hCount);
        #endif
        break;
    case 1:
    case 4:
    case 5:
    case 6:
    case 7:
        for (unsigned int i = 0, off = 0; i < m_wSeg; off += m_wCounts[i++]) {
            _UpdateBuf3D(m_s,
                m_sBuf, m_sBufH, m_sBufW, m_hOff, m_wOffs[i], m_dOff,
                m_rBuf + off * m_rBufH, m_rBufH, m_rBufW, 'h',
                m_hCount, m_wCounts[i], m_dCount,
                "variables");
        }
        break;
    case 2:
        _UpdateTex(m_s,
            *m_s->GetTTexA(m_pos), *m_s->GetTTexB(m_pos), m_hOff, m_wOffs[0],
            m_rBuf, m_rBufH, 'h',
            m_hCount, m_wCounts[0],
            "texture constants");
        break;
    case 3:
        _UpdateTex(m_s,
            *m_s->GetCTexA(m_pos), *m_s->GetCTexB(m_pos), m_hOff, m_wOffs[0],
            m_rBuf, m_rBufH, 'h',
            m_hCount, m_wCounts[0],
            "common constants");
        break;
    case 8:
        m_s->SetIterNo((unsigned int)_FloatAsInt(*m_rBuf));
        break;
    }

}

/**********************************************************************************************************************/

INLINEH unsigned int _Roll(int c, unsigned int roll, unsigned int size) {
    c -= roll;
    return (c < 0) ? c + size : c;
}

INLINE unsigned int _Unroll(unsigned int c, unsigned int roll, unsigned int size) {
    c += roll;
    return (c >= size) ? c - size : c;
}

/**********************************************************************************************************************/

// Currently unused.  May be used in future to speed up coordinate transform.

INLINE void _DivMod(int &a, unsigned int b, int &d) {
    d = a / b;
    a -= d * b;
}

INLINE void _ShiftMod(int &a, unsigned int b, int &d) {
    d = a >> b;
    a &= (1 << b) - 1;
}

/**********************************************************************************************************************/

#include "common_def.h"

#include "util_def.h"

#include _USER_GLOBAL_DEF

#include _SYNC_DEF

#include "kernel_macros.h"

#include _USER_ALLKERNELS_DEF
