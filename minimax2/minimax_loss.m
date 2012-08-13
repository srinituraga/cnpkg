function [dloss,loss,classerr,randIndex] = minimax_loss(bmap,segTrue,margin,pos)
% MINIMAX_LOSS Compute the minimax image segmentation loss function and its derivative

% format the inputs
bmap = single(bmap);
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
[dloss,loss,classerr,randIndex] = minimax_loss_mex(bmap,segTrue,margin,pos);
