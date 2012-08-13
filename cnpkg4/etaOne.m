function m = etaOne(m)
% set the eta (learning rates) for all the parameters to one

for l = 1:length(m.layers),
    type = [m.package '_' m.layers{l}.type];
    classes = superclasses(type); classes{end+1} = type;
    if isfield(m.layers{l},'eta'),
        m.layers{l}.eta = 1;
    end
end
