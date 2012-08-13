for k=1:length(th),
	for j=1:length(nets),
		disp(['Bean counting net ' nets{j} ' at threshold ' num2str(th(k)) '...'])
		eval(['[compEst,compEstSz] = connectedComponents(conn' nets{j} '>th(k));']);
		[ri(k,j),stats] = randIndex(compTrue,compEst);
		tpr(k,j) = stats.truePosRate; fpr(k,j) = stats.falsePosRate;
		prec(k,j) = stats.prec; rec(k,j) = stats.truePosRate;
		splt(k,j) = stats.splitRate; mrg(k,j) = stats.mergeRate;
% 		lut = 1:length(compEstSz); lut(compEstSz<=20) = 0;
% 		compEst = applylutWithZeros(compEst,lut);
		[splits(k,j),mergers(k,j),missing(k,j)] = split_merge_counts_bipartite(compTrue,compEst,0,0);
	end
end
