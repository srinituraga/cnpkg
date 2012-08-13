f1=figure(10);
clf(f1)
f2=figure(11);
clf(f2)

figure(f1)
subplot(141)
set(gca,'LineStyleOrder','.-|.--','NextPlot','replacechildren')
plot(train.th,train.ri,'.--')
hold on
plot(test.th,test.ri,'.-')
hold off
ylim([.925 1])
grid on
%legend(train.legendStr{:},test.legendStr{:},'Location','SouthWest');
xlabel('Threshold')
ylabel('Fraction correct')
title('A. Rand index')

subplot(142)
set(gca,'LineStyleOrder','.-|.--','NextPlot','replacechildren')
loglog(train.fpr,train.tpr,'.--')
hold on
plot(test.fpr,test.tpr,'.-')
hold off
grid on
%legend(train.legendStr{:},test.legendStr{:},'Location','SouthEast');
xlabel('FPR [fp/(fp+tn)]')
ylabel('TPR [tp/(tp+fn)]')
title('B. ROC curve')

subplot(143)
set(gca,'LineStyleOrder','.-|.--','NextPlot','replacechildren')
plot(train.rec,train.prec,'.--')
hold on
plot(test.rec,test.prec,'.-')
hold off
grid on
%legend(train.legendStr{:},test.legendStr{:},'Location','SouthWest');
xlabel('Recall [tp/(tp+fn)]')
ylabel('Precision [tp/(tp+fp)]')
title('C. Precision-Recall curve')

subplot(144)
set(gca,'LineStyleOrder','.-|.--','NextPlot','replacechildren')
plot(train.splits,train.mergers,'.--')
hold on
plot(test.splits,test.mergers,'.-')
hold off
grid on
legend(train.legendStr{:},test.legendStr{:});
xlabel('Splits/object')
ylabel('Mergers/object')
title('D. Splits vs. Mergers')
axis([0 5 0 1])


figure(f2)
subplot(211)
set(gca,'LineStyleOrder','.-|.--','NextPlot','replacechildren')
plot(train.th,train.splt,'.--')
hold on
plot(test.th,test.splt,'.-')
hold off
%ylim([.925 1])
grid on
legend(train.legendStr{:},test.legendStr{:},'Location','SouthWest');
xlabel('Threshold')
ylabel('fn/(fn+tn)')
title('False negatives (Splits)')
subplot(212)
set(gca,'LineStyleOrder','.-|.--','NextPlot','replacechildren')
plot(train.th,train.mrg,'.--')
hold on
plot(test.th,test.mrg,'.-')
hold off
%ylim([.925 1])
grid on
legend(train.legendStr{:},test.legendStr{:},'Location','SouthWest');
xlabel('Threshold')
ylabel('fp/(fp+tp)')
title('False positives (Mergers)')

% figure(9)
% set(gca,'LineStyleOrder','.-|.--','NextPlot','replacechildren')
% plot(train.splits,train.missing,'.--')
% hold on
% plot(test.splits,test.missing,'.-')
% hold off
% grid on
% legend(train.legendStr{:},test.legendStr{:},'Location','SouthWest');
% xlabel('Splits')
% ylabel('Missing')
% title('Splits vs. Missing')
% axis([0 5 0 5])
