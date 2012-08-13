function [inout,stats] = compute_rand_index(set,nets,scales,offset,thresholds)
% read all segmentations corresponding to a training or test set

if ~exist('scales','var') || isempty(scales),
	scales = [1];
end
if ~exist('offset','var') || isempty(offset),
	offset = [0 0];
end

if ~exist('thresholds','var') || isempty(thresholds),
	thresholds = [0.1:0.02:0.6];
end

type = 'color';
%testing_file = ['/home/sturaga/net_sets/cnpkg/BSD/' set '_' type '_boundary'];
%load(['/home/sturaga/data/BSDS300/' set '_' type '.mat'],'im','seg')
testing_file = ['/home/sturaga/net_sets/cnpkg/BSD/bsds500_' set];
load(['/home/sturaga/net_sets/BSD/minimax/bsds500_' set],'im','seg')


% Process all the networks
for k = 1:length(nets),
	needToLoad = true;
	while needToLoad,
		try,
			load(['/home/sturaga/saved_networks/' num2str(nets(k)) '/latest.mat']);
			needToLoad = false;
		catch,
			disp(['Failed to load ' '/home/sturaga/saved_networks/' num2str(nets(k)) '/latest.mat'])
			disp('retrying in 5 seconds...')
			pause(5)
		end
	end
	msave{k} = m;

	% catch and fix up if using an old 'm'
	if ~isfield(m.layer_map,'label'),
		m=cnpkg_mknet_init(m);
	end
	if isfield(m.layer_map,'big_input'),
		m.layers{m.layer_map.big_input}.inputblock{1} = [];
	end

	inout{k} = inout_test_multiscale(m,testing_file,scales,true,true);
end

nh = mknhood2d(1);

n_ri(length(im)) = 0;
ri = zeros(length(im),length(thresholds),length(nets));
tp = zeros(length(im),length(thresholds),length(nets));
fp = zeros(length(im),length(thresholds),length(nets));
tn = zeros(length(im),length(thresholds),length(nets));
fn = zeros(length(im),length(thresholds),length(nets));
ri_best = zeros(length(im),length(thresholds),length(nets));
for iFrame = 1:length(im),
	n_ri(iFrame) = length(seg{iFrame});
	disp(['Processing frame ' num2str(iFrame) '...'])
	for iThreshold = 1:length(thresholds),
		for iNet = 1:length(nets),

% 			conn = inout2conn(squeeze(inout{iNet}{iFrame}),nh);
% 			segEst = connectedComponents(conn>thresholds(iThreshold),nh,50);
% 			segEst = markerWatershed(conn,nh,segEst,segEst==0,0);

			segEst = watershed(imhmin(1-squeeze(inout{iNet}{iFrame}),thresholds(iThreshold),4),4);


			for iSeg = 1:length(seg{iFrame}),
				segTrue = seg{iFrame}{iSeg};% .* mask{iFrame}{iSeg};
				[ri_iSeg,stats] = randIndex(segTrue( ...
								offset(1)+1:end-offset(1), ...
								offset(2)+1:end-offset(2)), ...
								segEst( ...
								offset(1)+1:end-offset(1), ...
								offset(2)+1:end-offset(2)));
				disp(['    seg ' num2str(iSeg) ': ' num2str(ri_iSeg)]);
				ri(iFrame,iThreshold,iNet) = ri(iFrame,iThreshold,iNet) + ri_iSeg;
				tp(iFrame,iThreshold,iNet) = tp(iFrame,iThreshold,iNet) + stats.truePos;
				fp(iFrame,iThreshold,iNet) = fp(iFrame,iThreshold,iNet) + stats.falsePos;
				tn(iFrame,iThreshold,iNet) = tn(iFrame,iThreshold,iNet) + stats.trueNeg;
				fn(iFrame,iThreshold,iNet) = fn(iFrame,iThreshold,iNet) + stats.falseNeg;
				ri_best(iFrame,iThreshold,iNet) = max(ri_best(iFrame,iThreshold,iNet),ri_iSeg);
			end
			ri(iFrame,iThreshold,iNet) = ri(iFrame,iThreshold,iNet)/length(seg{iFrame});

		end
	end
end

clear stats
stats.m = msave;
stats.tpr = squeeze(sum(tp)./sum(tp+fn));
stats.fpr = squeeze(sum(fp)./sum(fp+tn));
stats.prec = squeeze(sum(tp)./sum(tp+fp));
stats.rec = stats.tpr;
stats.ri_mean = squeeze(mean(ri,1));
stats.ri_best = squeeze(mean(ri_best,1));
stats.ri_best2 = squeeze(mean(max(ri_best,[],2),1));
stats.thresh = thresholds(:);
stats.legendStr = str2cell(num2str(nets(:)'));
% disp(['Final Rand index: ' num2str(stats.ri_mean)])

% mkdir(['/home/sturaga/exhibit/' net '/'])
% save(['/home/sturaga/exhibit/' net '/' set '_out'],'inout','ri','stats')

figure(10)
plot(stats.thresh,stats.ri_mean)
legend(stats.legendStr)
grid on
ylim([.7 .84])

end
