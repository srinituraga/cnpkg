COMPUTED_SENS_HANDLE hs = GET_COMPUTED_SENS_HANDLE(ZN);
int ySize = COMPUTED_SENS_HANDLE_Y_SIZE(hs);
int xSize = COMPUTED_SENS_HANDLE_X_SIZE(hs);
int dSize = COMPUTED_SENS_HANDLE_D_SIZE(hs);

int nSize = COMPUTED_SENS_HANDLE_N_SIZE(hs);

LAYER_VAL_HANDLE hv = GET_LAYER_VAL_HANDLE(ZP);
int vy1 = Y_SIZE - 1 - THIS_Y;
int vx1 = X_SIZE - 1 - THIS_X;
int vd1 = D_SIZE - 1 - THIS_D;

int f  = THIS_F;
int nf = THIS_NF;

float dw = 0.0f;

for (int n = 0; n < nSize; n++) {
for (int sd = 0, vd = vd1; sd < dSize; sd++, vd++) {
for (int sx = 0, vx = vx1; sx < xSize; sx++, vx++) {
for (int sy = 0, vy = vy1; sy < ySize; sy++, vy++) {

    float v = READ_LAYER_VAL_HANDLE(hv, f, vy, vx, vd, n);
    float s = READ_COMPUTED_SENS_HANDLE(hs, nf, sy, sx, sd, n);

    dw += v * s;

}
}
}
}

WRITE_DVAL(dw);

dw /= (ySize * xSize * dSize * nSize);
float w = READ_VAL;
w += ETA * dw;

WRITE_VAL(w);
