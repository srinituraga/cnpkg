COMPUTED_SENS_HANDLE hs = GET_COMPUTED_SENS_HANDLE(ZN);
int ySize = COMPUTED_SENS_HANDLE_Y_SIZE(hs);
int xSize = COMPUTED_SENS_HANDLE_X_SIZE(hs);
int dSize = COMPUTED_SENS_HANDLE_D_SIZE(hs);
int nSize = COMPUTED_SENS_HANDLE_N_SIZE(hs);

int nf = THIS_NF;

float db = 0.0f;

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


int normalizationFactor = ySize * xSize * dSize * nSize;
db /= normalizationFactor;
db /= normalizationFactor;
db *= nSize;

float b = READ_VAL;
b += ETA * db;

WRITE_VAL(b);
