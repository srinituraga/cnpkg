function [ri,stats] = ...
					randIndex(compTrue,compEst,restrictedRadius,radius)

% % create atomic regions for out space bits of true objects
% if exist('atomicOutSpace','var') && ~isempty(atomicOutSpace) && atomicOutSpace,
% 	compEstInZero = (compEst==0).*compTrue;
% 	idxNonZero = find(compEstInZero);
% 	compEstInZero(idxNonZero) = compEstInZero(idxNonZero) + max(compEst(:));
% 	compEst = compEst + compEstInZero;
% end

% compute basic numbers from pairs of pixels either
% restricted to a particular radius of separation (slow)
% or
% unrestricted (fast)
if exist('restrictedRadius','var') && ~isempty(restrictedRadius) && restrictedRadius,
	if ~exist('radius','var') || isempty(radius),
		radius = 10;
	end
	[nPairsTotal,nPosTrue,nPosActual,nPosCorrect] = randIndexRestricted(compTrue,compEst,radius);
else,
	[nPairsTotal,nPosTrue,nPosActual,nPosCorrect] = randIndexUnRestricted(compTrue,compEst);
end

% number of pairs classified as negative
nNegTrue = nPairsTotal - nPosTrue;
nNegActual = nPairsTotal - nPosActual;
nNegCorrect = nPairsTotal + nPosCorrect - nPosTrue - nPosActual;

% the number of incorrectly classified pairs
nPosIncorrect = nPosTrue-nPosCorrect;
nNegIncorrect = nNegTrue-nNegCorrect;
nPairsIncorrect = nPosIncorrect + nNegIncorrect;

% clustering error
stats.clusteringError = nPairsIncorrect/nPairsTotal;

% some stats
stats.falsePos = (nPosActual-nPosCorrect);
stats.falseNeg = (nNegActual-nNegCorrect);
stats.truePos = nPosCorrect;
stats.trueNeg = nNegCorrect;

% more derived stats
stats.truePosRate = stats.truePos/(stats.truePos+stats.falseNeg);
stats.falsePosRate = stats.falsePos/(stats.falsePos+stats.trueNeg);
stats.prec = stats.truePos/(stats.truePos+stats.falsePos);
stats.rec = stats.truePosRate;
stats.mergeRate = stats.falsePos/(stats.falsePos+stats.truePos);
stats.splitRate = stats.falseNeg/(stats.falseNeg+stats.trueNeg);

% rand index
ri = 1 - stats.clusteringError;

return

function [nPairsTotal,nPosTrue,nPosActual,nPosCorrect] = randIndexRestricted(compTrue,compEst,radius)

[imx,jmx,kmx] = size(compTrue);
nhood = mknhood2(radius);
nPairsTotal = 0;
nPosTrue = 0;
nPosActual = 0;
nPosCorrect = 0;

for nbor = 1:size(nhood,1),
	idxi = max(1-nhood(nbor,1),1):min(imx-nhood(nbor,1),imx);
	idxj = max(1-nhood(nbor,2),1):min(jmx-nhood(nbor,2),jmx);
	idxk = max(1-nhood(nbor,3),1):min(kmx-nhood(nbor,3),kmx);
	inPair = (compTrue(idxi,idxj,idxk) > 0) ...
		 & (compTrue(idxi+nhood(nbor,1),idxj+nhood(nbor,2),idxk+nhood(nbor,3)) > 0);
	posTrue = (compTrue(idxi,idxj,idxk) ...
		 == compTrue(idxi+nhood(nbor,1),idxj+nhood(nbor,2),idxk+nhood(nbor,3))) ...
	 & inPair;
	posEst = (compEst(idxi,idxj,idxk) ...
		 == compEst(idxi+nhood(nbor,1),idxj+nhood(nbor,2),idxk+nhood(nbor,3))) ...
	 & inPair;
	
	nPairsTotal = nPairsTotal + sum(inPair(:));
	nPosTrue = nPosTrue + sum(posTrue(:));
	nPosActual = nPosActual + sum(posEst(:));
	nPosCorrect = nPosCorrect + sum(posTrue(:)&posEst(:));
end

return


function [nPairsTotal,nPosTrue,nPosActual,nPosCorrect] = randIndexUnRestricted(compTrue,compEst)

% condition input (also, shift to make positive)
compTrue = double(compTrue)+1; maxCompTrue = max(compTrue(:));
compEst = double(compEst)+1; maxCompEst = max(compEst(:));

% compute the overlap or confusion matrix
% this computes the fraction of each true component
% overlapped by an estimated component
% the sparse() is used to compute a histogram over object pairs
overlap = sparse(compTrue(:),compEst(:),1,maxCompTrue,maxCompEst);

% compute the effective sizes of each set of objects
% computing it from the overlap matrix normalizes the sizes
% of each set of objects to the intersection of assigned space
% compTrueSizes = full(sum(overlap(2:end,:),2));
% compEstSizes = full(sum(overlap(2:end,2:end),1));
compTrueSizes = full(sum(overlap,2));
compEstSizes = full(sum(overlap,1));

% prune out the zero component (now 1, after shift) in the labeling (un-assigned "out" space)
% overlap = overlap(2:end,2:end);		% leads to a symmetric rand index
[compTrueIdx,compEstIdx,overlapSz] = find(overlap);

% total number of pairs
numPixTotal = sum(compTrueSizes);
nPairsTotal = numPixTotal*(numPixTotal-1)/2;

% the number of true positive
nPosTrue = sum(compTrueSizes.*(compTrueSizes-1))/2;

% the number of pairs actually classified as positive
nPosActual = sum(compEstSizes.*(compEstSizes-1))/2;

% the number of pairs correctly classified as positive
nPosCorrect = sum(overlapSz.*(overlapSz-1))/2;

return
