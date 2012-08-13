function struct2spreadsheet(filename, s, write_mode)
    fn = fieldnames(s);
    sfn = length(fn);
    sn = length(s.(fn{1}));
    
    
    if ~exist('write_mode','var') || strcmp(write_mode, 'w')
        fid = fopen(filename, 'w');
    else
        fid = fopen(filename, 'a');
    end
    outline = fn{1};
    for f = 2:sfn
        outline = [outline '\t' fn{f}]; 
    end
    outline = [outline '\n'];
    fprintf(fid, outline);
            
    fclose(fid);
        
        
    for n = 1:sn
        
        outline = data_fmt(s.(fn{1}){n});
        for f = 2:sfn
            outline = [outline '\t' data_fmt(s.(fn{f}){n})]; 
        end
        outline = [outline '\n'];
        
        fid = fopen(filename, 'a');
        fprintf(fid, outline);
        fclose(fid);
    end
    

end


function str = data_fmt(d)
    str = num2str(d);
end