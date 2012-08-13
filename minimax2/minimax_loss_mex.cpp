/* Matlab Notes:
 * mwSize and mwIndex are functionally equivalent to unsigned ints.
 * mxArray is a fortran-style N-d array basically containing
 * a linear array with meta-data describing the dimension sizes, etc
 */

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
class IndexCompare{
	private:
	const T * mValueArray;
	public:
		IndexCompare(const T * ValueArray){
			mValueArray = ValueArray;
		}
		bool operator() (const mwIndex& ind1, const mwIndex& ind2) const {
			return (mValueArray[ind1] > mValueArray[ind2]);
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
    // 3d boundary map [y * x * z]
	if (!mxIsSingle(prhs[0]))
		mexErrMsgTxt("bmap array must be floats (singles)");
	const mwSize ndims_bmap = mxGetNumberOfDimensions(prhs[0]);
	const mwSize* dims_bmap = mxGetDimensions(prhs[0]);
	mwSize M, N, P; M = dims_bmap[0]; N = dims_bmap[1]; P = 1;
	if (ndims_bmap == 3)
		P = dims_bmap[2];
	const float* bmap = (const float*) mxGetData(prhs[0]);

    // true target segmentation [y * x * z]
	if (!mxIsUint16(prhs[1]))
		mexErrMsgTxt("seg array must be uint16");
	const uint16_t* seg = (const uint16_t*)mxGetData(prhs[1]);

    // sq-sq loss margin [0.3]
    const double margin = (const double)mxGetScalar(prhs[2]);

    // is this a positive example pass [true] or a negative example pass [false] ?
    const bool pos = mxIsLogicalScalarTrue(prhs[3]);


	/* Output arrays */
    // the derivative of the MALIS-SqSq loss function
    // (times the derivative of the logistic activation function) [y * x * z * #edges]
    plhs[0] = mxCreateNumericArray(ndims_bmap,dims_bmap,mxSINGLE_CLASS,mxREAL);
    mxArray* dloss = plhs[0];
    float* dloss_data = (float*)mxGetData(dloss);

	/* Cache for speed to access neighbors */
	mwSize nPix = M*N*P;
	mwSize MN = M*N;
	int nbors;
	if (ndims_bmap==2) nbors = 4;
	else if (ndims_bmap==3) nbors = 6;
	else mexErrMsgTxt("Must be 2d or 3d!!");
	mwIndex nHoodIdx[6]= {1, -1, M, -M, MN, -MN};
	mwIndex nHoodSub[6][3] = {	{ 1, 0, 0},
								{-1, 0, 0},
								{ 0, 1, 0},
								{ 0,-1, 0},
								{ 0, 0, 1},
								{ 0, 0, -1} };

	/* Disjoint sets and the object pixel bag "overlap" */
	vector<map<mwIndex,mwIndex> > overlap(nPix);
	vector<mwIndex> rank(nPix);
	vector<mwIndex> parent(nPix);
	map<mwIndex,mwSize> segSizes;
	mwSize nLabeledPix=0;
    mwSize nPairPos=0;
	boost::disjoint_sets<mwIndex*, mwIndex*> dsets(&rank[0],&parent[0]);
	for ( mwIndex i = 0; i < nPix; ++i) {
		dsets.make_set(i);
		if (0 != seg[i]) {
			overlap[i].insert(pair<mwIndex,mwIndex>(seg[i],1));
			++nLabeledPix;
            ++segSizes[seg[i]];
            nPairPos += (segSizes[seg[i]]-1);
		}
	}

	mwSize nPairTot = (nLabeledPix*(nLabeledPix-1))/2;
    mwSize nPairNeg = nPairTot - nPairPos;
    mwSize nPairNorm;
    if (pos) {nPairNorm = nPairPos;} else {nPairNorm = nPairNeg;}

	/* Sort all the pixels in increasing order of weight */
	std::vector< mwIndex > pqueue( static_cast< mwSize > (nPix) );
	for ( mwIndex i = 0; i < nPix; ++i )
		pqueue[ i ] = i;
	sort( pqueue.begin(), pqueue.end(), IndexCompare<float>( bmap ) );

	/* Start MST */
	mwIndex minIdx, minSub[3];
	mwIndex nborIdx, nborSub[3];
	mwIndex set1, set2, tmp;
    mwSize nPair = 0;
	double loss=0, dl=0;
    mwSize nPairIncorrect = 0;
	map<mwIndex,mwIndex>::iterator it1, it2;

    /* Start flooding */
    for ( mwIndex i = 0; i < pqueue.size(); ++i ) {

    	// map linear idx
    	minIdx = pqueue[i];
    	set1 = dsets.find_set(minIdx);
		minIdx = pqueue[i];
		minSub[0] = (minIdx%MN)%M; minSub[1] = (minIdx%MN)/M; minSub[2] = minIdx/MN;

//mxAssert((bmap[minIdx]-1)==minIdx,"not equal!!");
//std::cerr << "minIdx: " << minIdx << ": " << "[" << minSub[0] << "," << minSub[1] << "," << minSub[2] << "] = " << bmap[minIdx] << std::endl;
    	for ( int j = 0; j < nbors; ++j) {

    		// bounds checking
    		bool OOB = false;
    		for (int k = 0; k < ndims_bmap; ++k) {
    			nborSub[k] = minSub[k] + nHoodSub[j][k];
    			OOB |= (nborSub[k]<0) || (nborSub[k]>=dims_bmap[k]);
    		}
    		if (OOB) continue;

    		nborIdx = minIdx+nHoodIdx[j];
			set2 = dsets.find_set(nborIdx);
			if ((bmap[nborIdx]>=bmap[minIdx]) && (set1!=set2)){
				dsets.link(set1, set2);

				/* compute the dloss for this MST edge */
				for (it1 = overlap[set1].begin();
						it1 != overlap[set1].end(); ++it1) {
					for (it2 = overlap[set2].begin();
							it2 != overlap[set2].end(); ++it2) {

	                    nPair = it1->second * it2->second;

						if (pos && (it1->first == it2->first)) {
	                        // +ve example pairs
	                        // Sq-Sq loss is used here
	                        dl = max(0.0,0.5+margin-bmap[minIdx]);
	                        loss += 0.5*dl*dl*nPair;
	                        dloss_data[minIdx] += dl*nPair;
	                        if (bmap[minIdx] <= 0.5) { // an error
	                            nPairIncorrect += nPair;
	                        }

						} else if ((!pos) && (it1->first != it2->first)) {
	                        // -ve example pairs
	                        // Sq-Sq loss is used here
							dl = -max(0.0,bmap[minIdx]-0.5+margin);
	                        loss += 0.5*dl*dl*nPair;
	                        dloss_data[minIdx] += dl*nPair;
	                        if (bmap[minIdx] > 0.5) { // an error
	                            nPairIncorrect += nPair;
	                        }
						}
					}
				}
	            dloss_data[minIdx] /= nPairNorm;

				/* move the pixel bags of the non-representative to the representative */
				if (dsets.find_set(set1) == set2) // make set1 the rep to keep and set2 the rep to empty
					swap(set1,set2);

				it2 = overlap[set2].begin();
				while (it2 != overlap[set2].end()) {
					it1 = overlap[set1].find(it2->first);
					if (it1 == overlap[set1].end()) {
						overlap[set1].insert(pair<mwIndex,mwIndex>(it2->first,it2->second));
					} else {
						it1->second += it2->second;
					}
					overlap[set2].erase(it2++);
				}
			} // end link

		}
	} // end while
// std::cerr << "Hellow!!" << std::endl;

    /* Return items */
    double classerr, randIndex;
    if (nlhs > 1) {
        loss /= nPairNorm;
        plhs[1] = mxCreateDoubleScalar(loss);
    }
    if (nlhs > 2) {
        classerr = (double)nPairIncorrect / (double)nPairNorm;
        plhs[2] = mxCreateDoubleScalar(classerr);
    }
    if (nlhs > 3) {
        randIndex = 1.0 - ((double)nPairIncorrect / (double)nPairNorm);
        plhs[3] = mxCreateDoubleScalar(randIndex);
    }
}
