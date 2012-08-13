#include "mex.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <queue>
#include <map>
using namespace std;

template <class T>
class AffinityGraphCompare{
	private:
	const T * mEdgeWeightArray;
	public:
		AffinityGraphCompare(const T * EdgeWeightArray){
			mEdgeWeightArray = EdgeWeightArray;
		}
		bool operator() (const mwIndex& ind1, const mwIndex& ind2) const {
			return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
		}
};



/*
 * Compute the MALIS loss function and its derivative wrt the affinity graph
 * MAXIMUM spanning tree
 * Author: Srini Turaga (sturaga@mit.edu)
 * All rights reserved
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

	/* input arrays */
    // 4d connectivity graph [y * x * z * #edges]
    const mxArray*	conn				= prhs[0];
	const mwSize	conn_num_dims		= mxGetNumberOfDimensions(conn);
	const mwSize*	conn_dims			= mxGetDimensions(conn);
	const mwSize	conn_num_elements	= mxGetNumberOfElements(conn);
	const float*	conn_data			= (const float*)mxGetData(conn);
    // graph neighborhood descriptor [3 * #edges]
	const mxArray*	nhood				= prhs[1];
	const mwSize	nhood_num_dims		= mxGetNumberOfDimensions(nhood);
	const mwSize*	nhood_dims			= mxGetDimensions(nhood);
	const double*	nhood_data			= (const double*)mxGetData(nhood);
    // true target segmentation [y * x * z]
    const mxArray*	seg					= prhs[2];
	const mwSize	seg_num_dims		= mxGetNumberOfDimensions(seg);
	const mwSize*	seg_dims			= mxGetDimensions(seg);
	const mwSize	seg_num_elements	= mxGetNumberOfElements(seg);
	const uint16_t*	seg_data			= (const uint16_t*)mxGetData(seg);
    // threshold [0.5]
    const double    threshold			= (const double)mxGetScalar(prhs[3]);
    // hinge loss margin [0.3]
    const double    margin              = (const double)mxGetScalar(prhs[4]);
    // is this a positive example pass [true] or a negative example pass [false] ?
    const bool      pos                 = mxIsLogicalScalarTrue(prhs[5]);
    const bool      neg                 = mxIsLogicalScalarTrue(prhs[6]);

	if (!mxIsSingle(conn)){
		mexErrMsgTxt("Conn array must be floats (singles)");
	}
	if (nhood_num_dims != 2) {
		mexErrMsgTxt("wrong size for nhood");
	}
	if ((nhood_dims[1] != (conn_num_dims-1))
		|| (nhood_dims[0] != conn_dims[conn_num_dims-1])){
		mexErrMsgTxt("nhood and conn dimensions don't match");
	}
	if (!mxIsUint16(seg)){
		mexErrMsgTxt("seg array must be uint16");
	}

    /* Matlab Notes:
     * mwSize and mwIndex are functionally equivalent to unsigned ints.
     * mxArray is an n-d array fortran-style "container" basically containing
     * a linear array with meta-data describing the dimension sizes, etc
     */

	/* Output arrays */
    // the derivative of the MALIS-SqSq loss function
    // [y * x * z * #edges]
    // plhs[0] = mxCreateNumericArray(conn_num_dims,conn_dims,mxSINGLE_CLASS,mxREAL);
    // mxArray* dloss = plhs[0];
    // float* dloss_data = (float*)mxGetData(dloss);
    plhs[0] = mxCreateNumericArray(conn_num_dims,conn_dims,mxSINGLE_CLASS,mxREAL);
    mxArray* dloss = plhs[0];
    float* dloss_data = (float*)mxGetData(dloss);

	/* Cache for speed to access neighbors */
	mwSize nVert = 1;
	for (mwIndex i=0; i<conn_num_dims-1; ++i)
		nVert = nVert*conn_dims[i];

	vector<mwSize> prodDims(conn_num_dims-1); prodDims[0] = 1;
	for (mwIndex i=1; i<conn_num_dims-1; ++i)
		prodDims[i] = prodDims[i-1]*conn_dims[i-1];

    /* convert n-d offset vectors into linear array offset scalars */
	vector<int32_t> nHood(nhood_dims[0]);
	for (mwIndex i=0; i<nhood_dims[0]; ++i) {
		nHood[i] = 0;
		for (mwIndex j=0; j<nhood_dims[1]; ++j) {
			nHood[i] += (int32_t)nhood_data[i+j*nhood_dims[0]] * prodDims[j];
		}
	}

	/* Disjoint sets and sparse overlap vectors */
	mwSize nLabeledVert=0;
    mwSize nPairPos=0;
	vector<map<mwIndex,mwIndex> > overlap(nVert);
	// vector<mwIndex> rank(nVert);
	// vector<mwIndex> parent(nVert);
	mwIndex rank[nVert];
	mwIndex parent[nVert];
	map<mwIndex,mwSize> segSizes;
	boost::disjoint_sets<mwIndex*, mwIndex*> dsets(&rank[0],&parent[0]);
	for (mwIndex i=0; i<nVert; ++i){
		dsets.make_set(i);
		if (0!=seg_data[i]) {
			overlap[i].insert(pair<mwIndex,mwIndex>(seg_data[i],1));
			++nLabeledVert;
            ++segSizes[seg_data[i]];
            nPairPos += (segSizes[seg_data[i]]-1);
		}
	}
	mwSize nPairTot = (nLabeledVert*(nLabeledVert-1))/2;
    mwSize nPairNeg = nPairTot - nPairPos;
    mwSize nPairNorm = 0;
    double fracPos = 0;
    for ( map<mwIndex,mwSize>::iterator it = segSizes.begin();
    									it != segSizes.end(); ++it) {
    	fracPos += ((double)it->second-1.0)/((double)2.0*it->second);
    }
    double fracNeg = ((double)(segSizes.size() * (segSizes.size()-1)))/2;
    double fracNorm = 0;
    if (pos) nPairNorm += nPairPos;
    if (neg) nPairNorm += nPairNeg;
    if (pos) fracNorm += fracPos;
    if (neg) fracNorm += fracNeg;
// cout << "nPairPos " << nPairPos << endl;
// cout << "nPairNeg " << nPairNeg << endl;

	/* Sort all the edges in increasing order of weight */
	std::vector< mwIndex > pqueue( static_cast< mwSize > (conn_num_elements) );
	mwIndex j = 0, nextidx, idx[3];
	bool oob;
	for ( mwIndex e = 0, i = 0; e < conn_dims[3]; ++e )
		for ( idx[2] = 0; idx[2] < conn_dims[2]; ++idx[2] )
			for ( idx[1] = 0; idx[1] < conn_dims[1]; ++idx[1] )
				for ( idx[0] = 0; idx[0] < conn_dims[0]; ++idx[0], ++i )
				{
					// check bounds
					oob = false;
					for (int coord=0; coord < 3; ++coord) {
			            nextidx = idx[coord]+(mwIndex)nhood_data[e + coord*nhood_dims[0]];
			            oob |= (nextidx < 0) || (nextidx >= conn_dims[coord]);
			        }
					if ( !oob )
						pqueue[ j++ ] = i;
				}
	pqueue.resize(j);
	sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( conn_data ) );
// cout << conn_data[pqueue.front()] << endl;
// cout << "e: " << pqueue.front()/nVert << ", v1: " << pqueue.front()%nVert << ", v2: " << pqueue.front()%nVert+nHood[pqueue.front()/nVert] << endl;
// cout << conn_data[pqueue.back()] << endl;
// cout << "e: " << pqueue.back()/nVert << ", v1: " << pqueue.back()%nVert << ", v2: " << pqueue.back()%nVert+nHood[pqueue.back()/nVert] << endl;

	/* Start MST */
	mwIndex minEdge;
	mwIndex e, v1, v2;
	mwIndex set1, set2, tmp;
    mwSize nVert1, nVert2;
    mwIndex seg1, seg2;
	double loss=0, l=0;
    double frac = 0;
    double fracFalseNeg = 0, fracFalsePos = 0, fracTrueNeg = 0, fracTruePos = 0;
	map<mwIndex,mwIndex>::iterator it1, it2;

    /* Start Kruskal's */
    for ( mwIndex i = 0; i < pqueue.size(); ++i ) {
		minEdge = pqueue[i];
		e = minEdge/nVert; v1 = minEdge%nVert; v2 = v1+nHood[e];

		set1 = dsets.find_set(v1);
		set2 = dsets.find_set(v2);
		if (set1!=set2){
// if ((overlap[set1].size()>0) && (overlap[set2].size()>0))
// 	cout << "sz1: " << overlap[set1].size() << ", sz2:" << overlap[set2].size() << endl;
			dsets.link(set1, set2);
			// dloss_data[minEdge]=1;

			/* compute the dloss for this MST edge */
			for (it1 = overlap[set1].begin();
					it1 != overlap[set1].end(); ++it1) {
				for (it2 = overlap[set2].begin();
						it2 != overlap[set2].end(); ++it2) {

					seg1 = it1->first;
					seg2 = it2->first;
					nVert1 = it1->second;
					nVert2 = it2->second;
                    frac = ((double) (nVert1*nVert2))/((double) (segSizes[seg1]*segSizes[seg2]));
                    if (frac > 1) cout << frac << endl;

                    // +ve example pairs
					if (pos && (seg1 == seg2)) {
                        if (conn_data[minEdge] <= threshold) { // an error
                            fracFalseNeg += frac;
                        } else {
                        	fracTruePos += frac;
                        }

                        // hinge loss is used here
                        l = max(0.0,margin-(conn_data[minEdge]-threshold));
                        loss += frac * l;
                        dloss_data[minEdge] += frac * (l > 0.0);
					}
                    // -ve example pairs
					if (neg && (seg1 != seg2)) {
                        if (conn_data[minEdge] > threshold) { // an error
                            fracFalsePos += frac;
                        } else {
                        	fracTrueNeg += frac;
                        }
                        // hinge loss is used here
                        l = max(0.0,margin+(conn_data[minEdge]-threshold));
                        loss += frac * l;
                        dloss_data[minEdge] -= frac * (l > 0.0);
					}
// cout << "   [" << seg1 << "," << seg2 << "]: " << nVert1 << "*" << nVert2 << "=" << nPair << endl;
				}
			}
// cout << endl;
            // dloss_data[minEdge] /= nPairNorm;
            dloss_data[minEdge] /= fracNorm;

			/* move the pixel bags of the non-representative to the representative */
			if (dsets.find_set(set1) == set2) // make set1 the rep to keep and set2 the rep to empty
				swap(set1,set2);

			for (it2 = overlap[set2].begin();
					it2 != overlap[set2].end(); ++it2) {
				it1 = overlap[set1].find(it2->first);
				if (it1 == overlap[set1].end()) {
					overlap[set1].insert(pair<mwIndex,mwIndex>(it2->first,it2->second));
				} else {
					it1->second += it2->second;
				}
			}
			overlap[set2].clear();
		} // end link

	} // end while

// cout << "pos , neg" << endl;
// cout << nPairFalsePos+nPairTruePos << " , " << nPairFalseNeg+nPairTrueNeg << endl;
// cout << "( fp , fn , tp , tn )" << endl;
// cout << "( " << nPairFalsePos << " , " << nPairFalseNeg << " , " << nPairTruePos << " , " << nPairTrueNeg <<  " )" << endl;

    /* Return items */
    if (nlhs > 1) {
        // loss /= nPairNorm;
        loss /= fracNorm;
        plhs[1] = mxCreateDoubleScalar(loss);
    }
    double* d;
    if (nlhs > 2) {
    	plhs[2] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    	d = (double*) mxGetData(plhs[2]);
    	d[0] = fracFalsePos;
        d[0] /= fracNorm;
    }
    if (nlhs > 3) {
    	plhs[3] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    	d = (double*) mxGetData(plhs[3]);
    	d[0] = fracFalseNeg;
        d[0] /= fracNorm;
    }
    if (nlhs > 4) {
    	plhs[4] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    	d = (double*) mxGetData(plhs[4]);
    	d[0] = fracTruePos;
        d[0] /= fracNorm;
    }
    if (nlhs > 5) {
    	plhs[5] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    	d = (double*) mxGetData(plhs[5]);
    	d[0] = fracTrueNeg;
        d[0] /= fracNorm;
    }
}
