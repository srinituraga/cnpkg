function zs = cns_grouplayers(m, g, num)

if nargin < 3, num = inf; end

zs = [];

if ~isfield(m, 'layers'), return; end

for z = 1 : numel(m.layers)
    if isfield(m.layers{z}, 'groupNo') && (m.layers{z}.groupNo == g)
        zs(end + 1) = z;
        if numel(zs) >= num, break; end
    end
end

return;