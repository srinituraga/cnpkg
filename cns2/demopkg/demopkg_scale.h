// Compute kernel for a "scale" cell.  A port of MATLAB's imresize using bicubic interpolation.
// Note: this kernel is separable and could therefore be rewritten as two stages.

#BLOCKSIZE 16 8

// Compute scaling factors.
float pySpace = LAYER_Y_SPACE(PZ);
float pxSpace = LAYER_X_SPACE(PZ);
float yFactor = pySpace / Y_SPACE; // If less than 1 we are shrinking.
float xFactor = pxSpace / X_SPACE;

// Compute receptive field size.
int yWidth = (yFactor < 1) ? (int)ceilf(4.0f / yFactor) : 4;
int xWidth = (xFactor < 1) ? (int)ceilf(4.0f / xFactor) : 4;

// Find receptive field index ranges.
int vy1, vy2, vx1, vx2; // Clipped to fall within the image ("valid").
int y1, y2, x1, x2;     // Unclipped, may go off image edges.
FIND_LAYER_Y_NEAREST(PZ, yWidth, vy1, vy2, y1, y2);
FIND_LAYER_X_NEAREST(PZ, xWidth, vx1, vx2, x1, x2);

// Determine internal filter coordinates.
float yFiltSpace = (yFactor < 1) ? yFactor : 1.0f;
float xFiltSpace = (xFactor < 1) ? xFactor : 1.0f;
float yFiltStart = (LAYER_Y_CENTER(PZ, y1) - THIS_Y_CENTER) / pySpace * yFiltSpace;
float xFiltStart = (LAYER_X_CENTER(PZ, x1) - THIS_X_CENTER) / pxSpace * xFiltSpace;

VAL_HANDLE h = GET_LAYER_VAL_HANDLE(PZ);
float num = 0.0f; // Result numerator.
float den = 0.0f; // Result denominator.

float xFilt = xFiltStart;
for (int x = x1; x <= x2; x++, xFilt += xFiltSpace) {

    // Compute x component of filter.
    float fx = fabsf(xFilt);
    float wx;
    if (fx <= 1.0f) {
        wx = (1.5f * fx - 2.5f) * fx * fx + 1.0f;
    } else if (fx <= 2.0f) {
        wx = ((-0.5f * fx + 2.5f) * fx - 4.0f) * fx + 2.0f;
    } else {
        wx = 0.0f;
    }

    // Determine x index of image pixel to read.
    int vx = (x < vx1) ? vx1 : (x > vx2) ? vx2 : x;

    float yFilt = yFiltStart;
    for (int y = y1; y <= y2; y++, yFilt += yFiltSpace) {

        // Compute y component of filter.
        float fy = fabsf(yFilt);
        float wy;
        if (fy <= 1.0f) {
            wy = (1.5f * fy - 2.5f) * fy * fy + 1.0f;
        } else if (fy <= 2.0f) {
            wy = ((-0.5f * fy + 2.5f) * fy - 4.0f) * fy + 2.0f;
        } else {
            wy = 0.0f;
        }

        // Determine y index of image pixel to read.
        int vy = (y < vy1) ? vy1 : (y > vy2) ? vy2 : y;

        num += wy * wx * READ_VAL_HANDLE(h, 0, vy, vx);
        den += wy * wx;

    }
}

// Clip output to [0 1].
WRITE_VAL(fminf(fmaxf(num / den, 0.0f), 1.0f));
