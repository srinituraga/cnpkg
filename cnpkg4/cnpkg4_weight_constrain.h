// Block size is usually [16 N] where N is found by experiment.  See CNS manual.
// N might need to be reduced for cards below GTX 285/295.
#BLOCKSIZE 16 16

// Apply constraints and update the weights

float yWPxSize = Y_SPACE;
float xWPxSize = X_SPACE;
float dWPxSize = D_SPACE;

float yOffset = floorf(THIS_Y/yWPxSize)*yWPxSize;
float xOffset = floorf(THIS_X/xWPxSize)*xWPxSize;
float dOffset = floorf(THIS_D/dWPxSize)*dWPxSize;

float dw = 0;
for (float d = dOffset; d < (dOffset+dWPxSize); d++) {
for (float x = xOffset; x < (xOffset+xWPxSize); x++) {
for (float y = yOffset; y < (yOffset+yWPxSize); y++) {
    dw += READ_WEIGHT_DVAL(THIS_Z,THIS_F,(int) y,(int) x,(int) d,THIS_NF);
}
}
}
dw /= yWPxSize*xWPxSize*dWPxSize;

float w;
if (!GRADONLY) {
    w = READ_VAL;
    w -= GLOBALETA * ETA * dw;
    WRITE_VAL(w);
} else {
    WRITE_DVAL(dw);
}
