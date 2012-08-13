function []=cnpkg2_split_process(m, testing_set, output_hdf5_file, output_hdf5_path, split_size, ignore_bb)

if ischar(m),
	load(m,'m')
end

if ~exist('split_size','var') || isempty(split_size),
	split_size = [1 1 1]*175;
end

% load data
if ischar(testing_set),
	testing_set = load(testing_set);
end
disp(['Processing im_file: ' testing_set.im_file ', im_path: ' testing_set.im_path])
testing_set

% create output file
[numdims dims maxdims]=get_hdf5_size(testing_set.im_file,testing_set.im_path);
create_hdf5_file(output_hdf5_file,output_hdf5_path,[dims(1:3) m.params.output_units],[split_size m.params.output_units],[split_size m.params.output_units],'float');
if ~isfield(testing_set,'bb') || (exist('ignore_bb','var') && ignore_bb),
	testing_set.bb = [1 1 1;dims(1:3)]';
end

% initialize on gpu
for k=1:3,
	split_size(k) = min(split_size(k),testing_set.bb(k,2)-testing_set.bb(k,1)+1);
end
m = cnpkg2_mapdim_layers_fwd(m,split_size,1);
fprintf(['Initializing...']);tic
cns('init',m)
fprintf(' done. ');toc

% generate split points
block_bb = double(generate_splitpoints(testing_set.bb, split_size, m.offset));
if m.params.input_units>1,
	block_bb(4,1,:) = 1;
	block_bb(4,2,:) = m.params.input_units;
end
nBlock = size(block_bb,3);

% go through each split box for this image
for iBlock=1:nBlock,

	fprintf(['Processing block ' num2str(iBlock) '/' num2str(nBlock) '...']);tic	
	% read input from disk
	fprintf([' reading...']);
	input = get_hdf5_file(testing_set.im_file,testing_set.im_path,block_bb(:,1,iBlock),block_bb(:,2,iBlock));
	input = permute(input,[4 1 2 3]);

	% run the fwd pass
	fprintf([' computing...']);
	cns('set',{m.layer_map.input,'val',input});
	output = cns('step',m.step_map.fwd(1)+1,m.step_map.fwd(end),{m.layer_map.output,'val'});
	output = permute(output,[2 3 4 1]);

	% write output to disk
	fprintf([' writing...']);
	write_hdf5_file(output_hdf5_file,output_hdf5_path,[block_bb(:,1,iBlock)'+m.offset 1], ...
								[block_bb(:,2,iBlock)'-m.offset m.params.output_units],output);
	fprintf(' done. ');toc
end


cns done

return
