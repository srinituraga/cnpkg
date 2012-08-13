function netinfo = browseNetworks(netfolder, fields, lbl, prepcmd, range)

ndir = dir(netfolder);
ndir(1:2)=[];

    
    if ~exist('lbl','var')
        lbl = [];
    end
    if ~exist('prepcmd','var')
        prepcmd = [];
    end
    
    for nn = length(lbl)+1:length(fields)
        lbl{nn} = ['var' num2str(nn)];
    end
    
for nn = 1:length(fields)
    eval(['netinfo.' lbl{nn} ' = [];']);
end
disp([ num2str(length(ndir)) ' files and directories found']);

for nn = 1:length(ndir);
    if exist('range','var') && ~any(str2double(ndir(nn).name)==range);
        continue
    end
    
    try
        mfiles = dir([netfolder ndir(nn).name]);
        mfiles(1:2)=[];
        
        
        
        
        epochs = zeros(length(mfiles),1);
        for l = 1:length(mfiles)
            if ~isempty(str2double(mfiles(l).name(7:end-4)))
                epochs(l)=str2double(mfiles(l).name(7:end-4));
            end
        end
        [~, maxepind] = max(epochs);
        %         disp(numepochs);
        clear n
        load([netfolder ndir(nn).name '/' mfiles(maxepind).name]);
        if exist('n','var')
            m=n;
        end
        
        
    catch ME
        disp(ME.message);
        disp(['dir name ' ndir(nn).name ' is no good :<']);
        
        continue
        
    end
    
    
    for evaln = 1:length(prepcmd)
        try
            eval(prepcmd{evaln});
        catch ME
            disp(ME.message);
            disp(['preparatory command ' num2str(evaln) ' failed']);
        end
    end
    for evaln = 1:length(fields)
        try
            eval(['netinfo.' lbl{evaln} '{end+1} = ' fields{evaln} ';']);
        catch
            eval(['netinfo.' lbl{evaln} '{end+1} = -1;']);
        end
    end
    
end
% 
% for nn = 1:length(fields)
%     try
%         eval(['netinfo.var' num2str(nn) ' = cell2mat(netinfo.var' num2str(nn) ');']);
%     catch
%     end
% end

end