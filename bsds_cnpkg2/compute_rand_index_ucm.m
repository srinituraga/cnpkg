function [inout,stats,thresholds] = compute_rand_index(set,net,scales,thresholds)
% read all segmentations corresponding to a training or test set

if isnumeric(net),
	net = num2str(net);
end

if ~exist('scales','var') || isempty(scales),
	scales = [1 1/2 1/4 1/8];
end

if ~exist('thresholds','var') || isempty(thresholds),
	thresholds = [0.4:0.02:0.98];
end

type = 'color';
load(['/home/sturaga/saved_networks/' net '/latest.mat']);
if ~isfield(m.layer_map,'label'),
	m=cnpkg_mknet_init(m);
end
m.data_info.testing_file = ['/home/sturaga/net_sets/cnpkg/BSD/' set '_' type '_boundary'];
load(['/home/sturaga/data/BSDS300/' set '_' type '.mat'],'im','seg','mask')

if isfield(m.layer_map,'big_input'),
	m.layers{m.layer_map.big_input}.inputblock{1} = [];
end

inout = inout_test_multiscale(m,scales,1);
for iFrame = 1:length(inout),
	inout{iFrame} = 1-single(inout2ucm(squeeze(inout{iFrame}),'imageSize'))/255;
end
offset = m.layers{m.layer_map.label}.offset;

nh = mknhood2d(1);

n_ri(length(im)) = 0;
ri = zeros(length(im),length(thresholds));
tp = zeros(length(im),length(thresholds));
fp = zeros(length(im),length(thresholds));
tn = zeros(length(im),length(thresholds));
fn = zeros(length(im),length(thresholds));
ri_best = zeros(length(im),length(thresholds));
for iFrame = 1:length(im),
	n_ri(iFrame) = length(seg{iFrame});
	disp(['Processing frame ' num2str(iFrame) '...'])
	for iThreshold = 1:length(thresholds),

		conn = inout2conn(inout{iFrame},nh);
		segEst = connectedComponents(conn>thresholds(iThreshold),nh,50);
		segEst = markerWatershed(conn,nh,segEst,segEst==0,0);

		for iSeg = 1:length(seg{iFrame}),
			segTrue = seg{iFrame}{iSeg};% .* mask{iFrame}{iSeg};
			[ri_iSeg,stats] = randIndex(segTrue( ...
							offset(1)+1:end-offset(1), ...
							offset(2)+1:end-offset(2)), ...
							segEst);
			disp(['    seg ' num2str(iSeg) ': ' num2str(ri_iSeg)]);
			ri(iFrame,iThreshold) = ri(iFrame,iThreshold) + ri_iSeg;
			tp(iFrame,iThreshold) = tp(iFrame,iThreshold) + stats.truePos;
			fp(iFrame,iThreshold) = fp(iFrame,iThreshold) + stats.falsePos;
			tn(iFrame,iThreshold) = tn(iFrame,iThreshold) + stats.trueNeg;
			fn(iFrame,iThreshold) = fn(iFrame,iThreshold) + stats.falseNeg;
			ri_best(iFrame,iThreshold) = max(ri_best(iFrame,iThreshold),ri_iSeg);
		end
		ri(iFrame,iThreshold) = ri(iFrame,iThreshold)/length(seg{iFrame});

	end
end

clear stats
stats.tpr = sum(tp)./sum(tp+fn);
stats.fpr = sum(fp)./sum(fp+tn);
stats.prec = sum(tp)./sum(tp+fp);
stats.rec = stats.tpr;
stats.ri_mean = mean(ri,1);
stats.ri_best = mean(ri_best,1);
disp(['Final Rand index: ' num2str(stats.ri_mean)])

mkdir(['/home/sturaga/exhibit/' net '/'])
save(['/home/sturaga/exhibit/' net '/' set '_out_ucm'],'inout','ri','ri_best')

end
