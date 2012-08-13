COMPUTED_SENS_HANDLE hs = GET_COMPUTED_SENS_HANDLE(ZN);
int ySize = COMPUTED_SENS_HANDLE_Y_SIZE(hs);
int xSize = COMPUTED_SENS_HANDLE_X_SIZE(hs);
int dSize = COMPUTED_SENS_HANDLE_D_SIZE(hs);

LAYER_VAL_HANDLE hv = GET_LAYER_VAL_HANDLE(ZP);
int vy1 = Y_SIZE - 1 - THIS_Y;
int vx1 = X_SIZE - 1 - THIS_X;
int vd1 = D_SIZE - 1 - THIS_D;

int f  = THIS_F;
int nf = THIS_NF;

float r = 0.0f;

for (int sd = 0, vd = vd1; sd < dSize; sd++, vd++) {
for (int sx = 0, vx = vx1; sx < xSize; sx++, vx++) {
for (int sy = 0, vy = vy1; sy < ySize; sy++, vy++) {

    float v = READ_LAYER_VAL_HANDLE(hv, f, vy, vx, vd);
    float s = READ_COMPUTED_SENS_HANDLE(hs, nf, sy, sx, sd);

    r += v * s;

}
}
}
r /= (ySize * xSize * dSize);

WRITE_DVAL(r);

float w = READ_VAL;

w += ETA * r;

WRITE_VAL(w);
