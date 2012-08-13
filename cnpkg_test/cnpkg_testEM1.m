% load data
load ('/Users/viren/exhibit/0723/e1088_inout_validate4.mat');
input = im;
offset = 6;
output = labels;
outputMask = label_mask;

%%load Initial Bias & weigh values
load NetworkParam.mat

p = struct;
p.fCount = cell2mat (paramInit.num_of_maps_in_layer); %p.fCount  = [1 5 5 1];
p.fSize = [0 1 1 1] * paramInit.filter_size; %p.fSize   = [0 5 5 5];
p.fDepth  = [0 1 1 1] * paramInit.filter_size; %p.fDepth  = [0 5 5 5];
p.eta     = [0 1 1 1] * 1e-1; %TBD
p.outSize = [5 5 5 2];%p.outSize = paramInit.minibatch_size; %p.outSize = [5 5 5];
p.iterations = paramInit.training_iterations;

p.fullOutSize = size (squeeze (output));
m = cnpkg_buildmodel(p);

%cns('test', m, 'gpu', 'mean');
cns('init', m, 'gpu', 'mean');
%cns('init', m, 'cpu'); % run in emulation mode

%% Loading weights and biases from already trained matlab networks
% The weights in previous code were reversed because they were using
% ND convolution in place of correlation. The below code does the
% re-ordering

%loading weights
weight2 = zeros (5, 5, 5, 5);
for i = 1:5
    weight2(:,:,:,i) = cell2mat (paramInit.init_weights.W(2,i));
end

weight3 = zeros (5, 5, 5, 5, 5);
for i = 1:5
    w = cell2mat (paramInit.init_weights.W(3,i));
    for j=1:5
        w1 = w (:,:,:, j);
        weight3(5-j+1,:,:,:, i) = w1;
    end
end

weight4 = zeros (5, 5, 5, 5);
paramWeight4 = cell2mat (paramInit.init_weights.W(4));
for i=1:5
    weight4(5-i+1,:,:,:, 1) = paramWeight4(:,:,:,i);
end

% loading Biases
bias2 = cell2mat (paramInit.init_weights.B(2,:));
bias2 = bias2';
bias3 = cell2mat (paramInit.init_weights.B(3,:));
bias3 = bias3';
bias4 = cell2mat (paramInit.init_weights.B(4,1));
bias4 = bias4';

indexes = zeros(3, p.iterations, p.outSize(4));
indexes(1, :) = randi(p.fullOutSize(1) - m.layers{2}.size{2}, 1, p.iterations, p.outSize(4)) - 1;
indexes(2, :) = randi(p.fullOutSize(2) - m.layers{2}.size{3}, 1, p.iterations, p.outSize(4)) - 1;
indexes(3, :) = randi(p.fullOutSize(3) - m.layers{2}.size{4}, 1, p.iterations, p.outSize(4)) - 1;

%% Setting weights and biases
cns ('set', 3, 'val', shiftdim (weight2, -1));
cns ('set', 4, 'val', bias2);
cns ('set', 6, 'val', weight3);
cns ('set', 7, 'val', bias3);
cns ('set', 9, 'val', weight4);
cns ('set', 10, 'val', bias4);


%% Setting input, label, mask and index blocks
cns('set', 2, 'input', input);
cns('set', 11, 'labelblock', output);
cns('set', 1, 'val', indexes);
cns('set', 11, 'maskblock', outputMask);

%% Running the network step by step to verify accuracy. 
% The output.h kernel has to be changed so that the labels and masks are
% not read from the label and mask blocks but read from the label and mask
% patches in the output layer itself
test = 1;
if (test == 1)
    inputInfos = shiftdim (infos.training_input_patch, -1);
    inputInfos(:,:,:,:, 2) = shiftdim (infos.training_input_patch, -1);
    cns ('set', 2, 'val', inputInfos);
    cns('step', 2);
    cns('step', 3);
    cns('step', 4);
    outputInfos = shiftdim (infos.training_label_patch, -1);
    outputInfos(:,:,:,:,2) = shiftdim (infos.training_label_patch, -1);
    cns ('set', 11, 'labelpatch', outputInfos);
    outputMaskInfos = shiftdim (infos.training_gradient_mask_patch, -1);
    outputMaskInfos(:,:,:,:,2) = shiftdim (infos.training_gradient_mask_patch, -1);
    cns ('set', 11, 'maskpatch', single (outputMaskInfos));
    cns('step', 5);
    cns('step', 6);
    cns('step', 7);
    cns('step', 8);
else
    cns('run', p.iterations);
end

%% Network Accuracy verification code.
% Getting the parameters and values from the network
weight22 = cns ('get', 3, 'val'); weight22 = shiftdim (weight22, 1);
weight32 = cns ('get', 6, 'val'); weight32 = squeeze (weight32);
weight42 = cns ('get', 9, 'val'); weight42 = squeeze (weight42);
dweight22 = cns ('get', 3, 'dval'); dweight22 = shiftdim (dweight22, 1);
dweight32 = cns ('get', 6, 'dval'); dweight32 = squeeze (dweight32);
dweight42 = cns ('get', 9, 'dval'); dweight42 = squeeze (dweight42);
bias22 = cns ('get', 4, 'val');
bias32 = cns ('get', 7, 'val');
bias42 = cns ('get', 10, 'val');
dbias22 = cns ('get', 4, 'dval');
dbias32 = cns ('get', 7, 'dval');
dbias42 = cns ('get', 10, 'dval');
act1 = cns ('get', 2, 'val');
act2 = cns ('get', 5, 'val');
act3 = cns ('get', 8, 'val');
act4 = cns ('get', 11, 'val');
sens2 = cns ('get', 5, 'sens');
sens3 = cns ('get', 8, 'sens');
sens4 = cns ('get', 11, 'sens');

% Getting weights from paramTrained for comparison
dweight22Info = zeros (5, 5, 5, 5);
weight22Info = zeros (5, 5, 5, 5);
for i = 1:5
    dweight22Info(:,:,:,i) = cell2mat (paramTrained.weights.gradientW(2,i));
    weight22Info(:,:,:,i) = cell2mat (paramTrained.weights.W(2,i));
end

dweight32Info = zeros (5, 5, 5, 5, 5);
weight32Info = zeros (5, 5, 5, 5, 5);
for i = 1:5
    w = cell2mat (paramTrained.weights.W(3,i));
    dw = cell2mat (paramTrained.weights.gradientW(3,i));
    for j=1:5
        w1 = w (:,:,:, j);
        dw1 = dw (:,:,:, j);
        weight32Info(5-j+1,:,:,:, i) = w1;
        dweight32Info(5-j+1,:,:,:, i) = dw1;
    end
end

weight42Info = zeros (5, 5, 5, 5);
dweight42Info = zeros (5, 5, 5, 5);
paramWeight4 = cell2mat (paramTrained.weights.W(4));
dparamWeight4 = cell2mat (paramTrained.weights.gradientW(4));
for i=1:5
    weight42Info(5-i+1,:,:,:, 1) = paramWeight4(:,:,:,i);
    dweight42Info(5-i+1,:,:,:, 1) = dparamWeight4(:,:,:,i);
end

%Getting activity and sensitivity from layers
act11 = shiftdim (infos.activity{1}, -1);
sens11 = shiftdim (infos.sensitivity{1}, -1);

act22 = zeros (5, 13, 13, 13);
for i=1:5
    act22(i,:,:,:) = infos.activity{2}(:,:,:,i);
end
sens22 = zeros (5, 13, 13, 13);
for i=1:5
    sens22(i,:,:,:) = infos.sensitivity{2}(:,:,:,i);
end

act32 = zeros (5, 9, 9, 9);
for i=1:5
    act32(i,:,:,:) = infos.activity{3}(:,:,:,i);
end
sens32 = zeros (5, 9, 9, 9);
for i=1:5
    sens32(i,:,:,:) = infos.sensitivity{3}(:,:,:,i);
end

act42 = shiftdim (infos.activity{4}, -1);
sens42 = shiftdim (infos.sensitivity{4}, -1);

dGradW2 = dweight22 - dweight22Info;
dGradW3 = dweight32 - dweight32Info;
dGradW4 = dweight42 - dweight42Info;
dW2 = weight22 - weight22Info;
dW3 = weight32 - weight32Info;
dW4 = weight42 - weight42Info;
% dAct2 = act2 - act22;
% dAct3 = act3 - act32;
% dAct4 = act4 - act42;
% dSens2 = sens2 - sens22;
% dSens3 = sens3 - sens32;
% dSens4 = sens4 - sens42;

cns done;