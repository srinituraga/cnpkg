train = load('metrics_e2006_roi2');
test = load('metrics_e2006_roi5');
% train = load('metrics_validate4');
% test = load('metrics_roi7');

idxkeeptrain = 1:length(train.legendStr);
idxkeeptest = 1:length(test.legendStr);
%idxkeeptrain = []; train.th = [];

trainStr = train.legendStr(idxkeeptrain);
%trainStr = {'Standard (Train)','Minimax (Train)'};
testStr = test.legendStr(idxkeeptest);
%testStr = {'Standard (Test)','Minimax (Test)'};

% throw away unnecessary networks
train.legendStr = trainStr;
train.ri = train.ri(:,idxkeeptrain);
train.tpr = train.tpr(:,idxkeeptrain);
train.fpr = train.fpr(:,idxkeeptrain);
train.rec = train.rec(:,idxkeeptrain);
train.prec = train.prec(:,idxkeeptrain);
train.splt = train.splt(:,idxkeeptrain);
train.mrg = train.mrg(:,idxkeeptrain);
train.splits = train.splits(:,idxkeeptrain);
train.mergers = train.mergers(:,idxkeeptrain);
train.missing = train.missing(:,idxkeeptrain);

test.legendStr = testStr;
test.ri = test.ri(:,idxkeeptest);
test.tpr = test.tpr(:,idxkeeptest);
test.fpr = test.fpr(:,idxkeeptest);
test.rec = test.rec(:,idxkeeptest);
test.prec = test.prec(:,idxkeeptest);
test.splt = test.splt(:,idxkeeptest);
test.mrg = test.mrg(:,idxkeeptest);
test.splits = test.splits(:,idxkeeptest);
test.mergers = test.mergers(:,idxkeeptest);
test.missing = test.missing(:,idxkeeptest);

plot_metrics_train_test

% sz = 7; fnt = 'Helvetica';% 'Times';
% f=figure(10);
% set(f,'Renderer','painters')
% set(f,'PaperSize',[8 2])
% set(f,'PaperPosition',[-0.75 0 9.6 2])
% a=subplot(141);
% axis([0.5 1 0.93 1])
% grid off
% set(a,'FontSize',sz,'FontName',fnt);
% a=subplot(142);
% grid off
% set(a,'FontSize',sz,'FontName',fnt);
% a=subplot(143);
% grid off
% set(a,'FontSize',sz,'FontName',fnt);
% a=subplot(144);
% grid off
% set(a,'FontSize',sz,'FontName',fnt);
% legend boxoff
% axis([0 5 0 1])

if exist('exportpdf','var') && exportpdf==true,
	print(f,'-dpdf','/home/sturaga/tmp/roi8_roi7_NIPS.pdf')
end
