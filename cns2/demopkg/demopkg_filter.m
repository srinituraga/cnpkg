classdef demopkg_filter < demopkg_base
methods (Static)

%-----------------------------------------------------------------------------------------------------------------------

function f = CNSFields

f.fVals = {'ga', 'private', 'cache', 'dims', {1 2 1}, 'dparts', {1 1 2}, 'dnames', {'y' 'x' 'f'}};

end

%-----------------------------------------------------------------------------------------------------------------------

function m = InitLayer(m, z)

m = cns_super(m, z);

c = m.layers{z};

switch c.fParams{1}
case 'gabor', c.fVals = GenerateGabor(c.rfCount, c.size{1}, c.fParams{2 : end});
otherwise   , error('invalid filter type');
end

for f = 1 : c.size{1}
    a = c.fVals(:, :, f);
    a = a - mean(a(:));
    a = a / sqrt(sum(a(:) .* a(:)));
    c.fVals(:, :, f) = a;
end

m.layers{z} = c;

end

%-----------------------------------------------------------------------------------------------------------------------

end
end

%***********************************************************************************************************************

function fVals = GenerateGabor(rfCount, fCount, aspectRatio, lambda, sigma)

fVals = zeros(rfCount, rfCount, fCount);

points = (1 : rfCount) - ((1 + rfCount) / 2);

for f = 1 : fCount

    theta = (f - 1) / fCount * pi;

    for j = 1 : rfCount
        for i = 1 : rfCount

            x = points(j) * cos(theta) - points(i) * sin(theta);
            y = points(j) * sin(theta) + points(i) * cos(theta);

            if sqrt(x * x + y * y) <= rfCount / 2
                e = exp(-(x * x + aspectRatio * aspectRatio * y * y) / (2 * sigma * sigma));
                e = e * cos(2 * pi * x / lambda);
            else
                e = 0;
            end

            fVals(i, j, f) = e;

        end
    end

end

end