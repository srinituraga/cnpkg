// Kernel for a single cell of a bias layer.

// Block size is usually [16 N] where N is found by experiment.  See CNS manual.
// N might need to be reduced for cards below GTX 285/295.
#BLOCKSIZE 16 16

float db = 0.0f;
int ySize, xSize, dSize, nSize;

NODE_VAL_HANDLE hs = GET_NODE_VAL_HANDLE(ZS(0));
ySize = NODE_VAL_HANDLE_Y_SIZE(hs);
xSize = NODE_VAL_HANDLE_X_SIZE(hs);
dSize = NODE_VAL_HANDLE_D_SIZE(hs);
nSize = NODE_VAL_HANDLE_N_SIZE(hs);

int nf = THIS_NF;

for (int n = 0; n < nSize; n++) {
for (int d = 0; d < dSize; d++) {
for (int x = 0; x < xSize; x++) {
for (int y = 0; y < ySize; y++) {

    db -= READ_NODE_VAL_HANDLE(hs, nf, y, x, d, n);

}
}
}
}

WRITE_DVAL(db);

float b = READ_VAL;

if (!GRADONLY) {
    db /= ySize * xSize * dSize * nSize;

b -= GLOBALETA * ETA * db;
WRITE_VAL(b);

}
