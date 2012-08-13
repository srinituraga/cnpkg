% test malis_loss

tiny = 1e-4;
conn2 = conn1;
[dloss,loss] = malis_loss2(conn1,nh,comp1,0.1,true,true);
% dloss = int32(dloss);
dlossest = zeros(size(dloss));
% dlossest = zeros(size(dloss),'int32');
for k=1:numel(conn1),
	conn2(k) = conn1(k) + tiny;
	[~,loss2] = malis_loss2(conn2,nh,comp1,0.1,true,true);
	dlossest(k) = (loss-loss2)/tiny;
	conn2(k) = conn1(k);
	% if (abs(dloss(k))>eps) || (abs(dlossest(k))>eps)
	if (dloss(k)~=0) || (dlossest(k)~=0)
	% if (dloss(k)~=0)
		disp(['[' num2str(k) '] ' num2str([dloss(k) dlossest(k)])])
		% disp(['[' num2str(k) '] ' num2str([dloss(k) dlossest(k)*(abs(dlossest(k))>1e-5)])])
	end
end