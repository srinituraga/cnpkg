% EMNET: Parallel implementation of convolutional networks
% Developed and maintained by Viren Jain <viren@mit.edu>
% Do not distribute without permission.

% generate a list of split points for training or testing
% bb - bounding box to respect in input space
% sample_size - how large should each sample should be in input space
% input_indent - # of INPUT pixels the network loses due to valid convs

function [splitpoints]=generate_splitpoints(bb, sample_size, input_indent)

for i=1:length(input_indent)
	if(2*input_indent(i)>=sample_size(i))
		error('sample_size not big enough for network input_indent');
		splitpoints=0;
		return
	elseif(sample_size(i)>(bb(i,2)-bb(i,1)+1))
		error('sample_size bigger than bounding box bb');
		return
	end
end

prealloc_size=10000;
splitpoints=zeros(3,2,prealloc_size,'uint32');
num=1;

ptr=bb(:,1);
ptr_end=bb(:,2);

while ptr(1)<=bb(1,2)-(2*input_indent(1))
	%num
    ptr(2)=bb(2,1);
    while ptr(2)<=bb(2,2)-(2*input_indent(2))
        ptr(3)=bb(3,1);
        while ptr(3)<=bb(3,2)-(2*input_indent(3))
			
            % compute end of sample box
            for i=1:3
                ptr_end(i)=ptr(i)+sample_size(i)-1;

                % check if this sample exceeds labeled bounding box
                % if so, adjust so it just reaches the edge
                if(ptr_end(i)>bb(i,2))
                    ptr(i)=bb(i,2)-sample_size(i)+1;
                    ptr_end(i)=ptr(i)+sample_size(i)-1; % should equal bb(i,2)
                end
            end

            % add sample to list
            splitpoints(:,:,num)=[ptr, ptr_end];
			num=num+1;
			if(size(splitpoints,3)<num)
				new_splitpoints=zeros(3,2,size(splitpoints,3)+prealloc_size);
				new_splitpoints(:,:,1:end-prealloc_size)=splitpoints;
				splitpoints=new_splitpoints;
			end
			
            ptr(3)=ptr(3)+sample_size(3)-1-(2*input_indent(3))+1;
        end
        ptr(2)=ptr(2)+sample_size(2)-1-(2*input_indent(2))+1;
    end
    ptr(1)=ptr(1)+sample_size(1)-1-(2*input_indent(1))+1;
end

splitpoints=splitpoints(:,:,1:num-1);
