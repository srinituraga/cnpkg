function m = cnpkg_buildmodel(p)

m.package = 'cnpkg';
m.independent = true;

n = numel(p.fCount);

zw = [0, 2 + (0 : n - 2) * 3] + 1;
zb = [0, 3 + (0 : n - 2) * 3] + 1;
zx = [1, 4 + (0 : n - 2) * 3] + 1;

zi = 1;%layer number for indices for each iteration of learning

m.layers{zi}.name = 'xi';
m.layers{zi}.type = 'index';
m.layers{zi}.stepNo  = [];
m.layers{zi}.size = {3, p.iterations, p.outSize(4)};

m.layers{zx(1)}.z       = zx(1); % For display only.
m.layers{zx(1)}.name    = 'x1';
m.layers{zx(1)}.type    = 'input';
m.layers{zx(1)}.stepNo  = [1];
m.layers{zx(1)}.size{1} = 1;
m.layers{zx(1)}.zin = zi;
m.layers{zx(1)}.input = zeros (255, 255, 255);

for i = 2 : n - 1

    m.layers{zx(i)}.z       = zx(i); % For display only.
    m.layers{zx(i)}.name    = sprintf('x%u', i);
    m.layers{zx(i)}.type    = 'hidden';
    m.layers{zx(i)}.stepNo  = [i - 1, 2 * n - i] + 1;
    m.layers{zx(i)}.zp      = zx(i - 1);
    m.layers{zx(i)}.zw      = zw(i);
    m.layers{zx(i)}.zb      = zb(i);
    m.layers{zx(i)}.znw     = zw(i + 1);
    m.layers{zx(i)}.zn      = zx(i + 1);
    m.layers{zx(i)}.size{1} = p.fCount(i);

end

m.layers{zx(n)}.z       = zx(n); % For display only.
m.layers{zx(n)}.name    = sprintf('x%u', n);
m.layers{zx(n)}.type    = 'output';
m.layers{zx(n)}.stepNo  = [n - 1, n]  + 1;
m.layers{zx(n)}.zp      = zx(n - 1);
m.layers{zx(n)}.zw      = zw(n);
m.layers{zx(n)}.zb      = zb(n);
m.layers{zx(n)}.size{1} = p.fCount(n);
m.layers{zx(n)}.zin = zi;
m.layers{zx(n)}.labelblock = zeros (255, 255, 255);
m.layers{zx(n)}.maskblock = zeros (255, 255, 255);

for i = 2 : n

    m.layers{zw(i)}.z       = zw(i); % For display only.
    m.layers{zw(i)}.name    = sprintf('w%u', i);
    m.layers{zw(i)}.type    = 'weight';
    m.layers{zw(i)}.stepNo  = 2 * n;
    m.layers{zw(i)}.zp      = zx(i - 1);
    m.layers{zw(i)}.zn      = zx(i);
    m.layers{zw(i)}.eta     = p.eta(i);
    m.layers{zw(i)}.size{1} = p.fCount(i - 1);
    m.layers{zw(i)}.size{2} = p.fSize (i);
    m.layers{zw(i)}.size{3} = p.fSize (i);
    m.layers{zw(i)}.size{4} = p.fDepth(i);
    m.layers{zw(i)}.size{5} = p.fCount(i);

    m.layers{zb(i)}.z       = zb(i); % For display only.
    m.layers{zb(i)}.name    = sprintf('b%u', i);
    m.layers{zb(i)}.type    = 'bias';
    m.layers{zb(i)}.stepNo  = 2 * n;
    m.layers{zb(i)}.zn      = zx(i);
    m.layers{zb(i)}.eta     = p.eta(i);
    m.layers{zb(i)}.size{1} = p.fCount(i);
    m.layers{zb(i)}.size{2} = 1;

end

m = cns_mapdim(m, zx(n), 2, 'pixels', p.outSize(1));
m = cns_mapdim(m, zx(n), 3, 'pixels', p.outSize(2));
m = cns_mapdim(m, zx(n), 4, 'pixels', p.outSize(3));
m = cns_mapdim(m, zx(n), 5, 'pixels', p.outSize(4));

for i = n - 1 : -1 : 1

    m = cns_mapdim(m, zx(i), 2, 'int-td', zx(i + 1), p.fSize (i + 1), 1);
    m = cns_mapdim(m, zx(i), 3, 'int-td', zx(i + 1), p.fSize (i + 1), 1);
    m = cns_mapdim(m, zx(i), 4, 'int-td', zx(i + 1), p.fDepth(i + 1), 1);
    m = cns_mapdim(m, zx(i), 5, 'int-td', zx(i + 1), 1, 1);

end

m.layers{zx(n)}.offset = [(m.layers{zx(1)}.size{2} - p.outSize(1)) (m.layers{zx(1)}.size{3} - p.outSize(2)) (m.layers{zx(1)}.size{4} - p.outSize(3))]/2;

return;