function [dloss,loss,classerr,randIndex,falsePos,falseNeg,truePos,trueNeg] = ...
		 malis_loss(conn,nhood,segTrue,threshold,margin,pos,neg)
% MALIS_LOSS Compute the MaxiMin Affinity Learning for Image Segmentation loss function and its derivative

% format the inputs
conn = single(conn);
nhood = double(nhood);
segTrue = uint16(segTrue);
if ~exist('threshold','var') || isempty(threshold),
    threshold = 0.5;
else,
    threshold = double(threshold);
end
if ~exist('margin','var') || isempty(margin),
    margin = 0.3;
else,
    margin = double(margin);
end
if ~exist('pos','var') || isempty(pos),
    pos = true;
else,
    pos = logical(pos);
end
if ~exist('neg','var') || isempty(neg),
    neg = true;
else,
    neg = logical(neg);
end

% call the mex function
[dloss,loss,falsePos,falseNeg,truePos,trueNeg] = malis_loss_mex2(conn,nhood,segTrue,threshold,margin,pos,neg);
classerr = double(falsePos+falseNeg)/double(truePos+trueNeg+falsePos+falseNeg);
keyboard
randIndex = 1-classerr;
