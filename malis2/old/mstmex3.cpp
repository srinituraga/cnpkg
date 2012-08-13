#include "mex.h"
#include <iostream>
#include <cstdlib>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;


template <class T>
class AffinityGraph{
private:
    const T * mEdgeWeightArray;
public:
    AffinityGraph(const T * EdgeWeightArray){
        mEdgeWeightArray = EdgeWeightArray;
    }
    bool operator() (const mwIndex& ind1, const mwIndex& ind2) const
    {
        return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
    }
};



/*
 * MAXIMUM spanning tree
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    /* Input arrays */
    const mxArray*	conn				= prhs[0];
    const mwSize	conn_num_dims		= mxGetNumberOfDimensions(conn);
    const mwSize*	conn_dims			= mxGetDimensions(conn);
    const mwSize	conn_num_elements	= mxGetNumberOfElements(conn);
    const float*	conn_data			= (const float*)mxGetData(conn);
    if (!mxIsSingle(conn)){
        mexErrMsgTxt("Conn array must be floats (singles)");
    }

    /* Output arrays */
    plhs[0]			= mxCreateNumericArray(conn_num_dims,conn_dims,mxSINGLE_CLASS,mxREAL);
    float* mst_data	= (float*)mxGetData(plhs[0]);

    /* Cache for speed to access neighbors */
    mwSize nVert		= conn_dims[0]*conn_dims[1]*conn_dims[2];
    int32_t nHood[3]	= {-1, -conn_dims[0], -conn_dims[0]*conn_dims[1]};

    /* Disjoint sets and sparse overlap vectors */
    vector<mwIndex> rank(nVert);
    vector<mwIndex> parent(nVert);
    boost::disjoint_sets<mwIndex*, mwIndex*> dsets(&rank[0],&parent[0]);
    for (mwIndex i=0; i<nVert; ++i){
        dsets.make_set(i);
    }

    /* Priority queue for to sort all the edges in increasing order of weight */
    //priority_queue<mwIndex,vector<mwIndex>,AffinityGraph<float> > pqueue(conn_data);

    std::vector< mwIndex > pqueue( static_cast< mwSize >(3) *
                                   ( conn_dims[0]-1 ) *
                                   ( conn_dims[1]-1 ) *
                                   ( conn_dims[2]-1 ));

    mwIndex j = 0;

    for ( mwIndex d = 0, i = 0; d < 3; ++d )
        for ( mwIndex z = 0; z < conn_dims[2]; ++z )
            for ( mwIndex y = 0; y < conn_dims[1]; ++y )
                for ( mwIndex x = 0; x < conn_dims[0]; ++x, ++i )
                {
                    if ( x > 0 && y > 0 && z > 0 )
                        pqueue[ j++ ] = i;
                }

    std::sort( pqueue.begin(), pqueue.end(), AffinityGraph<float>( conn_data ) );


    /* Start MST */
    mwIndex minEdge;
    mwIndex e, v1;
    int32_t v2; // can become negative after adding nHood
    for ( mwIndex i = 0; i < pqueue.size(); ++i )
    {
        minEdge = pqueue[i];
        e = minEdge/nVert; v1 = minEdge%nVert; v2 = v1+nHood[e];

        //if ((v2 < 0) || (v2 >= nVert))
        //    continue;

        mwIndex set1=dsets.find_set(v1);
        mwIndex set2=dsets.find_set(v2);
        if (set1!=set2){
            dsets.link(set1, set2);
            mst_data[minEdge] = conn_data[minEdge];
        }
    }
}
