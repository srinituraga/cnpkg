COMPUTED_SENS_HANDLE hs = GET_COMPUTED_SENS_HANDLE(ZN);
int ySize = COMPUTED_SENS_HANDLE_Y_SIZE(hs);
int xSize = COMPUTED_SENS_HANDLE_X_SIZE(hs);
int dSize = COMPUTED_SENS_HANDLE_D_SIZE(hs);

int nf = THIS_NF;

float s = 0.0f;

for (int d = 0; d < dSize; d++) {
for (int x = 0; x < xSize; x++) {
for (int y = 0; y < ySize; y++) {

    s += READ_COMPUTED_SENS_HANDLE(hs, nf, y, x, d);

}
}
}

int normalizationFactor = ySize * xSize * dSize;
s /= normalizationFactor;
s /= normalizationFactor;

float b = READ_VAL;

b += ETA * s;

WRITE_DVAL(s);
WRITE_VAL(b);
