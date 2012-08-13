function plot_stats(net_array,legendstr)
% GUI stuff

fh = gcf; clf(fh);
pbh = uicontrol(fh,'Style','pushbutton','String','Update',...
    'Position',[1 1 50 20],'Callback',@updateDraw);

if ~exist('legendstr','var') || isempty(legendstr),
    legendstr = num2str(net_array(:));
end
for k=1:length(net_array),
    nets{k} = ['~/saved_networks/' num2str(net_array(k)) '/latest'];
end

updateDraw;

function updateDraw(varargin)
inet = 1; missing = [];
for k=1:length(nets),
    try,
        load(nets{k},'m');
        loss{2*inet-1} = 1:m.stats.epoch;
        loss{2*inet} = m.stats.loss(1:m.stats.epoch);
        classerr{2*inet-1} = 1:m.stats.epoch;
        classerr{2*inet} = m.stats.classerr(1:m.stats.epoch);
        inet=inet+1;
    catch,
        missing(end+1) = k;
        e = lasterror;
        disp(e.message)
    end
end
legendstr_tmp = legendstr;
legendstr_tmp(missing,:)=[];

ax(1)=subplot(121);
plot(loss{:})
legend(legendstr_tmp)
title('Loss')
xlabel('epoch')
grid on
ax(2)=subplot(122);
plot(classerr{:})
legend(legendstr)
title('Classification error')
xlabel('epoch')
grid on
linkaxes(ax,'x')
end

end
