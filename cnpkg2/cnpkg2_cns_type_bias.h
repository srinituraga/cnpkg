float db = 0.0f;
int ySize, xSize, dSize, nSize;

COMPUTED_SENS_HANDLE hs = GET_COMPUTED_SENS_HANDLE(ZN);
ySize = COMPUTED_SENS_HANDLE_Y_SIZE(hs);
xSize = COMPUTED_SENS_HANDLE_X_SIZE(hs);
dSize = COMPUTED_SENS_HANDLE_D_SIZE(hs);
nSize = COMPUTED_SENS_HANDLE_N_SIZE(hs);

int nf = THIS_NF;

for (int n = 0; n < nSize; n++) {
for (int d = 0; d < dSize; d++) {
for (int x = 0; x < xSize; x++) {
for (int y = 0; y < ySize; y++) {

    db += READ_COMPUTED_SENS_HANDLE(hs, nf, y, x, d, n);

}
}
}
}

WRITE_DVAL(db);


db /= ySize * xSize * dSize * nSize;

float b = READ_VAL;
b += GLOBALETA * ETA * db;

WRITE_VAL(b);
