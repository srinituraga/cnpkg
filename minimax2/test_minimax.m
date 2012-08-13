load /tmp/t
idx{1} = 64+[1:90]; idx{2} = 64+[1:90]; idx{3} = 64+6;
i = single(im(idx{:}));
s = seg(idx{:});
zdisp = ceil(size(i,3)/2);

bmap = i; bmap0 = bmap;
delta = 0;
margin = 1.00;
eta = 1e+1; lambda = 0.0;
nIter = 1e4;
l = zeros(nIter); c=l; r=l;
for k=1:nIter,
	[dloss,loss,classerr,randIndex] = minimax_loss(bmap,s,margin,true);
	l(k) = loss;
	c(k) = classerr;
	r(k) = randIndex;
	delta = lambda*delta + eta*dloss;
	[dloss,loss,classerr,randIndex] = minimax_loss(bmap,s,margin,false);
	l(k) = (l(k) + loss)/2;
	c(k) = (c(k) + classerr)/2;
	r(k) = (r(k) + randIndex)/2;
	delta = lambda*delta + eta*dloss;

	if norm(delta(:),'inf') < 1e-9, fprintf('converged!'), end
	bmap = bmap + delta;
	bmap = min(1,bmap); bmap = max(0,bmap);

	if ~rem(k,100),
		k0 = max(1,k-1000);
		figure(10), subplot(221), plot(l(k0:k))
		figure(10), subplot(222), plot(c(k0:k))
		figure(10), subplot(223), imagesc(bmap(:,:,zdisp))
		figure(10), subplot(224), imagesc(bmap(:,:,zdisp)-bmap0(:,:,zdisp))
		colormap gray
	end
end