% EMNET: Parallel implementation of convolutional networks
% Developed and maintained by Viren Jain <viren@mit.edu>
% Do not distribute without permission.

function [fid]=log_message(n, message)

message=['[',datestr(now,0),'] ',message];
disp(message)

[fid,msg]=fopen([n.params.save_string, 'log'], 'a+');
if fid<0,
	error(['failed to open ' n.params.save_string ' in log_message.m ' msg])
	rethrow(lasterror)
end

for i=1:size(message,1)
	fprintf(fid, [message(i,:), '\n']);
end
fclose(fid);

return
