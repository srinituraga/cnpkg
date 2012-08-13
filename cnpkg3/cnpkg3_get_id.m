% open a counter file, increment it, return new value
function [id]=get_id(counter_file)
counter=[];

fid=fopen(counter_file,'r+');
counter=fscanf(fid,'%i',[1 1]); 
counter=counter+1;
id=counter;
fseek(fid, 0, 'bof');
fprintf(fid,'%i',counter);
fclose(fid);
