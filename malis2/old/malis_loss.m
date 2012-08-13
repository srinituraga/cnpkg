function [dloss,loss,classerr,randIndex] = malis_loss(conn,nhood,segTrue,margin,pos)
% MALIS_LOSS Compute the MaxiMin Affinity Learning for Image Segmentation loss function and its derivative

% format the inputs
conn = single(conn);
nhood = double(nhood);
segTrue = uint16(segTrue);
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

% call the mex function
[dloss,loss,classerr,randIndex] = malis_loss_mex(conn,nhood,segTrue,margin,pos);
