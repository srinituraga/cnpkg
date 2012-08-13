function m = train(m,show_plot,platform,nice,skip_gpu_init)

if ischar(m),
	load(m,'m');
end
if exist('show_plot','var') && ischar(show_plot),
    show_plot = str2num(show_plot);
end

%% -----------------------Initializing
% configure IO
% configure save directory
if(isfield(m.params,'save_directory') && exist(m.params.save_directory,'dir'))
	m.params.save_string=[m.params.save_directory, num2str(m.ID),'/'];
	disp(['saving network to directory: ', m.params.save_string]); 
	mkdir(m.params.save_string);
else
	error('save directory not found!');
end

if ~exist('show_plot','var'),
	show_plot = 0;
end

% some minimax params
if ~isfield(m.params,'constrained_minimax'),
	m.params.constrained_minimax = true;
end

% initialize statistics
% initialize clocks
if(~isfield(m.stats, 'last_backup'))
	m.stats.last_backup=clock;
end

maxEpoch = ceil(m.params.maxIter/m.params.nIterPerEpoch);
if ~isfield(m.stats,'loss'),
	m.stats.loss = zeros(maxEpoch,1,'single');
	m.stats.classerr = zeros(maxEpoch,1,'single');
	m.stats.times = zeros(maxEpoch,1,'single');
else,
	m.stats.loss(maxEpoch) = 0;
	m.stats.classerr(maxEpoch) = 0;
	m.stats.times(maxEpoch) = 0;
end

% initialize random number generators
rand('state',sum(100*clock));
randn('state',sum(100*clock));

% init counters
if ~isfield(m.stats,'iter'),
	m.stats.iter = 0;
end
if ~isfield(m.stats,'epoch'),
	m.stats.epoch = 0;
end
cnpkg_log_message(m, ['initialized stats']);

% construct cns model for gpu
m = cnpkg4.MapDimFromOutput(m,m.params.output_size,2*m.params.minibatch_size);
[m, lastStep] = cnpkg4.SetupStepNo(m,1);
% totalBorder = 2*m.offset + cell2mat(m.layers{m.layer_map.output}.size(2:4)) - 1;
% halfTotalBorder = ceil(totalBorder/2);
totalBorder = cell2mat(m.layers{m.layer_map.input}.size(2:4)) - cell2mat(m.layers{m.layer_map.output}.size(2:4));
leftBorder = m.offset;
rightBorder = totalBorder - leftBorder;
halfOutputSize = floor(cell2mat(m.layers{m.layer_map.output}.size(2:4))/2);

% load data
training_set = load(m.data_info.training_file);
cnpkg_log_message(m,['Loaded data file...']);
% training_set.input = training_set.input(1:2);
% training_set.input2label = training_set.input2label(1:2);
% training_set.label = training_set.label(1:15);
% training_set.mask = training_set.mask(1:15);

if isfield(m.params,'zero_pad') && m.params.zero_pad,
	for k = 1:length(training_set.input),
		aa = training_set.input{k};
		for j = 1:4,
			sz(j) = size(aa,j);
		end
		training_set.input{k} = zeros(sz+[0 totalBorder],'single');
		training_set.input{k}(:, ...
			leftBorder(1)+(1:sz(1+1)), ...
			leftBorder(2)+(1:sz(1+2)), ...
			leftBorder(3)+(1:sz(1+3))) ...
				= aa;

		aa = training_set.label{k};
		for j = 1:4,
			sz(j) = size(aa,j);
		end
		training_set.label{k} = zeros(sz+[0 totalBorder],'single');
		training_set.label{k}(:, ...
			leftBorder(1)+(1:sz(1+1)), ...
			leftBorder(2)+(1:sz(1+2)), ...
			leftBorder(3)+(1:sz(1+3))) ...
				= aa;

		aa = training_set.mask{k};
		for j = 1:4,
			sz(j) = size(aa,j);
		end
		training_set.mask{k} = zeros(sz+[0 totalBorder],'single');
		training_set.mask{k}(:, ...
			leftBorder(1)+(1:sz(1+1)), ...
			leftBorder(2)+(1:sz(1+2)), ...
			leftBorder(3)+(1:sz(1+3))) ...
				= aa;
	end
end

nInput = length(training_set.input);
nLabel = length(training_set.label);
m.inputblock = training_set.input;
m.labelblock = training_set.label;
m.maskblock = training_set.mask;

% cache locations of positive and negative examples
for j = 1:nLabel,
    mask = training_set.mask{j}(:, ...
            (leftBorder(1)+halfOutputSize(1)+1):(end-rightBorder(1)-halfOutputSize(1)), ...
            (leftBorder(2)+halfOutputSize(2)+1):(end-rightBorder(2)-halfOutputSize(2)), ...
            (leftBorder(3)+halfOutputSize(3)+1):(end-rightBorder(3)-halfOutputSize(3))) > 0;

    if isequal(m.layers{m.layer_map.label}.type,'labelConn'),
        pos = permute(all(MakeConnLabel(permute(training_set.label{j}(:, ...
                (leftBorder(1)+halfOutputSize(1)+1):(end-rightBorder(1)-halfOutputSize(1)), ...
                (leftBorder(2)+halfOutputSize(2)+1):(end-rightBorder(2)-halfOutputSize(2)), ...
                (leftBorder(3)+halfOutputSize(3)+1):(end-rightBorder(3)-halfOutputSize(3))),[2 3 4 1])),4) > m.binarythreshold,[4 1 2 3]);
    else,
        pos = training_set.label{j}(:, ...
                (leftBorder(1)+halfOutputSize(1)+1):(end-rightBorder(1)-halfOutputSize(1)), ...
                (leftBorder(2)+halfOutputSize(2)+1):(end-rightBorder(2)-halfOutputSize(2)), ...
                (leftBorder(3)+halfOutputSize(3)+1):(end-rightBorder(3)-halfOutputSize(3))) > m.binarythreshold;
    end
    posIdx{j} = find(mask&pos);
    negIdx{j} = find(mask&~pos);
    labelSz{j} = size(mask);
end

m.layers{m.layer_map.minibatch_index}.size = {5,1,m.params.minibatch_size*2};
index = zeros(cell2mat(m.layers{m.layer_map.minibatch_index}.size));
m.layers{m.layer_map.minibatch_index}.val = index;

% setup weights to not normalize the gradient
for l = 1:length(m.layers),
    type = [m.package '_' m.layers{l}.type];
    classes = superclasses(type); classes{end+1} = type;
    if ismember([m.package '_weight'], classes) || ismember([m.package '_bias'], classes),
        m.layers{l}.gradonly = 1;
    end
end

% initialize gpu
if ~(exist('skip_gpu_init','var') && skip_gpu_init),
    cnpkg_log_message(m,'Initializing on device...');tic
    cns('init',m,platform,nice); m.inputblock = {[]}; m.labelblock = {[]}; m.maskblock = {[]};
    cnpkg_log_message(m,' done. ');toc
end

% if epoch 0, save
if m.stats.epoch == 0,
	m = cns('update',m);
	save([m.params.save_string,'epoch-' num2str(m.stats.epoch)],'m');
	save([m.params.save_string,'latest'],'m');
	m.stats.last_backup = clock;
end

% construct parameter vector
theta = [];
for l = 1:length(m.layers),
    type = [m.package '_' m.layers{l}.type];
    classes = superclasses(type); classes{end+1} = type;
    if ismember([m.package '_weight'], classes) || ismember([m.package '_bias'], classes),
        m.layers{l}.val = cns('get',l,'val');
        thetaNumel(l) = numel(m.layers{l}.val);
        theta = [theta;m.layers{l}.val(:)];
    end
end

opts.MaxIter=m.params.maxLineSearch;
opts.LS = 0;

    function [f,df] = grad(theta)
        df = zeros(size(theta));
        % set the parameters
        first = 0;
        for l = find(thetaNumel),
            m.layers{l}.val(:) = theta(first+[1:thetaNumel(l)]);
            cns('set',l,'val',m.layers{l}.val);
            first = first + thetaNumel(l);
        end

        % compute gradient
        [f,mask] = cns('step',[2 lastStep],{m.layer_map.error,'loss'},{m.layer_map.label,'mask'});
        summask = sum(mask(:));

        % get the loss & gradient
        f = cns('get',m.layer_map.error,'loss');
        f = sum(f(:).*mask(:))/summask;
        first = 0;
        for l = find(thetaNumel),
            df(first+[1:thetaNumel(l)]) = cns('get',l,'dval');
            first = first + thetaNumel(l);
        end
        df = df/summask;
    end

% done initializing!
cnpkg_log_message(m, ['initialization complete!']);
cns('set',0,'iter_no',1);



%% ------------------------Training
cnpkg_log_message(m, ['beginning training.. ']);
while(m.stats.epoch < maxEpoch),

    %% Timing -----------------------------------------------------------------------------------
    epoch_clock = clock;

    for epoch_iter = 1:m.params.nIterPerEpoch,
        %% Assemble minibatch indices ---------------------------------------------------------------
% if m.stats.iter==0,
fprintf('Assembling minibatch indices...');tic
        indexPos(4,1,:) = randi(nInput,[1 1 m.params.minibatch_size]);
        indexNeg(4,1,:) = randi(nInput,[1 1 m.params.minibatch_size]);

        % loop over images
        for k = 1:nInput,
            % pick a label image corresponding to this input image
            nLabel = length(training_set.input2label{k});
            idxLabelPos = (indexPos(4,:)==k);
            indexPos(5,idxLabelPos) = training_set.input2label{k}(randi(nLabel,[1 sum(idxLabelPos)]));
            idxLabelNeg = (indexNeg(4,:)==k);
            indexNeg(5,idxLabelNeg) = training_set.input2label{k}(randi(nLabel,[1 sum(idxLabelNeg)]));

            for j = training_set.input2label{k},
                InputLabel = find(idxLabelPos & (indexPos(5,:)==j));
                [jnk,indexPos(1,InputLabel),indexPos(2,InputLabel),indexPos(3,InputLabel)] = ind2sub( ...
                                labelSz{j},randsample(posIdx{j},length(InputLabel),1));
                InputLabel = find(idxLabelNeg & (indexNeg(5,:)==j));
                [jnk,indexNeg(1,InputLabel),indexNeg(2,InputLabel),indexNeg(3,InputLabel)] = ind2sub( ...
                                labelSz{j},randsample(negIdx{j},length(InputLabel),1));
            end
        end
        index(:,1,1:m.params.minibatch_size) = indexPos;
        index(:,1,m.params.minibatch_size+[1:m.params.minibatch_size]) = indexNeg;
        cns('set',{m.layer_map.minibatch_index,'val',index-1});
fprintf('done. '); toc


	%% Train ------------------------------------------------------------------------------------
% fprintf('Training on device...');tic
        cns('step',1);

        grad(theta);
        [mbloss,mbclasserr,mbmask] = cns('get',{m.layer_map.error,'loss'},{m.layer_map.error,'classerr'},{m.layer_map.label,'mask'});
        loss(epoch_iter) = sum(mbloss(:).*mbmask(:))/sum(mbmask(:));
        classerr(epoch_iter) = sum(mbclasserr(:).*mbmask(:))/sum(mbmask(:));


if m.params.debug, checkgrad(@grad,theta,1e-4); keyboard, end
        %[theta,f,nLineSearch] = minimize(@grad,theta,opts);
        [theta,f,nLineSearch] = m.params.minimizer(@grad,theta,opts);
    end
% fprintf('done. ');toc

    % copy weights back in for saving
    first = 0;
    for l = find(thetaNumel),
        m.layers{l}.val(:) = theta(first+[1:thetaNumel(l)]);
        first = first + thetaNumel(l);
    end

	%% Update counters --------------------------------------------------------------------------
	m.stats.iter = m.stats.iter+m.params.nIterPerEpoch;
	m.stats.epoch = m.stats.epoch+1;

	if m.params.debug >= 2,
		cnpkg_log_message(m,['DEBUG_MODE: loss: ' num2str(sum(loss(:)))]);
	end
	m.stats.loss(m.stats.epoch) = mean(loss(:));
	m.stats.classerr(m.stats.epoch) = mean(classerr(:));
    m.stats.times(m.stats.epoch) = etime(clock,epoch_clock);

	%% Save current state ----------------------------------------------------------------------
	if (etime(clock,m.stats.last_backup)>m.params.backup_interval),
		cnpkg_log_message(m, ['Saving network state... Iter: ' num2str(m.stats.iter)]);
		m.inputblock = {[]}; m.labelblock = {[]}; m.maskblock = {[]};
		for k=1:length(m.layers),
			switch m.layers{k}.type,
			case {'input', 'hidden', 'error', 'label'}
				if isfield(m.layers{k},'val'), m.layers{k}.val=0; end
				if isfield(m.layers{k},'sens'), m.layers{k}.sens=0; end
			end
		end
		save([m.params.save_string,'latest'],'m');
		m.stats.last_backup = clock;
	end

	if ~rem(m.stats.epoch,m.params.nEpochPerSave),
		%% save current state/statistics
		m.inputblock = {[]}; m.labelblock = {[]}; m.maskblock = {[]};
		for k=1:length(m.layers),
			switch m.layers{k}.type,
			case {'input', 'hidden', 'output', 'label'}
				if isfield(m.layers{k},'val'), m.layers{k}.val=0; end
				if isfield(m.layers{k},'sens'), m.layers{k}.sens=0; end
			end
		end
		save([m.params.save_string,'epoch-' num2str(m.stats.epoch)],'m');
		save([m.params.save_string,'latest'],'m');
		m.stats.last_backup = clock;
		cnpkg_log_message(m,['Epoch: ' num2str(m.stats.epoch) ', Iter: ' num2str(m.stats.iter) '; Loss ' num2str(mean(loss(:))) '; Classerr ' num2str(mean(classerr(:)))]);
	end

	% plot error stats
	if show_plot >= 1,
		try,
			figure(10)
			subplot(121)
			plot((1:m.stats.epoch)*m.params.nIterPerEpoch,m.stats.loss(1:m.stats.epoch))
			title('Loss')
			xlabel('iterations')
			subplot(122)
			plot((1:m.stats.epoch)*m.params.nIterPerEpoch,m.stats.classerr(1:m.stats.epoch))
			title('Classification error')
			xlabel('iterations')
			%plot(m.stats.loss(1:m.stats.iter))
			drawnow
		catch,
			warning('unable to plot')
		end
	end

	if show_plot >= 2 ,
		try,
			ii = cns('get',{m.layer_map.input,'val'});
			ii = permute(ii,[2 3 1 5 4]);
			ll = cns('get',{m.layer_map.label,'val'});
			ll = permute(ll,[2 3 1 5 4]);
			oo = cns('get',{m.layer_map.output,'val'});
			oo = permute(oo,[2 3 1 5 4]);
	[ll(:)';oo(:)']
			figure(2)
			smpl = randsample(m.params.minibatch_size*2,1,1);
			subplot(311), imagesc(ii(m.offset(1)+1:end-m.offset(1), ...
										m.offset(2)+1:end-m.offset(2), ...
										:,smpl))
			subplot(312), imagesc(ll(:,:,smpl),[0 1]),colormap gray
			subplot(313), imagesc(oo(:,:,smpl),[0 1]),colormap gray
			drawnow
		catch,
			warning('unable to plot')
		end
	end

end

m.inputblock = {[]}; m.labelblock = {[]}; m.maskblock = {[]};

end
