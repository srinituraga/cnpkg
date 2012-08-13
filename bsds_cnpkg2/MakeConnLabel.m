function conn = MakeConnLabel(cmp,nhood)
% Makes connectivity rep for arbitrary nhoods

[imx,jmx,kmx] = size(cmp);
conn = false([imx jmx kmx size(nhood,1)]);

for k = 1:size(nhood,1),
	idxi = max(1-nhood(k,1),1):min(imx-nhood(k,1),imx);
	idxj = max(1-nhood(k,2),1):min(jmx-nhood(k,2),jmx);
	idxk = max(1-nhood(k,3),1):min(kmx-nhood(k,3),kmx);
	conn(idxi,idxj,idxk,k) = ...
		(cmp(idxi,idxj,idxk) ...
		 == cmp(idxi+nhood(k,1),idxj+nhood(k,2),idxk+nhood(k,3))) ...
		& (cmp(idxi,idxj,idxk)~=0) ...
		& (cmp(idxi+nhood(k,1),idxj+nhood(k,2),idxk+nhood(k,3))~=0);
end
