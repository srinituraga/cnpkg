#BLOCKSIZE 16 8

// Find coordinates of input cells in the previous layer.

int y1, y2, x1, x2;
FIND_LAYER_Y_NEAREST(PZ, FVALS_Y_SIZE, y1, y2);
FIND_LAYER_X_NEAREST(PZ, FVALS_X_SIZE, x1, x2);

// Iterate over input cells.

float res = 0.0f;
float len = 0.0f;

for (int j = 0, x = x1; x <= x2; j++, x++) {
    for (int i = 0, y = y1; y <= y2; i++, y++) {

        // Read value of input cell.
        float v = READ_LAYER_VAL(PZ, 0, y, x);

        // Read corresponding filter value.
        float w = READ_FVALS(i, j, THIS_F);

        res += w * v;
        len += v * v;

    }
}

res = fabsf(res);
if (len > 0.0f) res /= sqrtf(len);

// Write out value of this cell.

WRITE_VAL(res);
