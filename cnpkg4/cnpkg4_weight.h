// Block size is usually [16 N] where N is found by experiment.  See CNS manual.
// N might need to be reduced for cards below GTX 285/295.
#BLOCKSIZE 16 16

// Compute the gradient

NODE_VAL_HANDLE hs = GET_NODE_VAL_HANDLE(ZS(0));
float ySize = NODE_VAL_HANDLE_Y_SIZE(hs);
float xSize = NODE_VAL_HANDLE_X_SIZE(hs);
float dSize = NODE_VAL_HANDLE_D_SIZE(hs);

float nSize = NODE_VAL_HANDLE_N_SIZE(hs);

NODE_VAL_HANDLE hv = GET_NODE_VAL_HANDLE(ZP);

float yRad = 0.5f * NODE_Y_SPACE(ZP) * Y_SIZE;
float xRad = 0.5f * NODE_X_SPACE(ZP) * X_SIZE;
float dRad = 0.5f * NODE_D_SPACE(ZP) * D_SIZE;

// first part computes the first 'val' index that's relevant to the first 'sens' index
// the second part is the shift corresponding to the weight that we are interested in
float vy1 = ((NODE_Y_START(ZS(0))-yRad-NODE_Y_START(ZP))/NODE_Y_SPACE(ZP)) + (Y_SIZE - 1 - THIS_Y);
float vx1 = ((NODE_X_START(ZS(0))-xRad-NODE_X_START(ZP))/NODE_X_SPACE(ZP)) + (X_SIZE - 1 - THIS_X);
float vd1 = ((NODE_D_START(ZS(0))-dRad-NODE_D_START(ZP))/NODE_D_SPACE(ZP)) + (D_SIZE - 1 - THIS_D);

// step the val index by this much
float deltavy = NODE_Y_SPACE(ZS(0))/NODE_Y_SPACE(ZP);
float deltavx = NODE_X_SPACE(ZS(0))/NODE_X_SPACE(ZP);
float deltavd = NODE_D_SPACE(ZS(0))/NODE_D_SPACE(ZP);

int f  = THIS_F;
int nf = THIS_NF;

float dw = 0.0f;

// the +0.5f is so that when vy gets cast to an integer,
// the implicit 'floor' will round to the nearest integer instead
float sd, sx, sy;
float vd, vx, vy;
float norm = 0.0f;
for (int n = 0; n < nSize; n++) {
for (sd = 0, vd = vd1+0.5f; sd < dSize; sd++, vd+=deltavd) {
for (sx = 0, vx = vx1+0.5f; sx < xSize; sx++, vx+=deltavx) {
for (sy = 0, vy = vy1+0.5f; sy < ySize; sy++, vy+=deltavy) {

    float v = READ_NODE_VAL_HANDLE(hv, f, (int) vy, (int) vx, (int) vd, n);
    float s = READ_NODE_VAL_HANDLE(hs, (int) nf, (int) sy, (int) sx, (int) sd, n);

    dw -= v * s;
    norm++;

}
}
}
}

if (!GRADONLY) {
    dw /= norm;
}
WRITE_DVAL(dw);
