// Block size is usually [16 N] where N is found by experiment.  See CNS manual.
// N might need to be reduced for cards below GTX 285/295.
#BLOCKSIZE 16 16

int f = THIS_F;
float ySpace = Y_SPACE;
float xSpace = X_SPACE;
float dSpace = D_SPACE;

int nOutputNodes = NUM_ZN;
float s = 0.0f;
for (int iOutput = 0; iOutput<nOutputNodes; iOutput++) {

    WEIGHT_VAL_HANDLE hw = GET_WEIGHT_VAL_HANDLE(ZNW(iOutput));
    int ySize = WEIGHT_VAL_HANDLE_Y_SIZE(hw);
    int xSize = WEIGHT_VAL_HANDLE_X_SIZE(hw);
    int dSize = WEIGHT_VAL_HANDLE_D_SIZE(hw);

    float yRad = 0.5f*ySize*ySpace;
    float xRad = 0.5f*xSize*xSpace;
    float dRad = 0.5f*dSize*dSpace;

    int y1, y2;
    int x1, x2;
    int d1, d2;
    GET_NODE_Y_RF_DIST(ZN(iOutput), yRad, y1, y2);
    GET_NODE_X_RF_DIST(ZN(iOutput), xRad, x1, x2);
    GET_NODE_D_RF_DIST(ZN(iOutput), dRad, d1, d2);

    float wyOffset = yRad-THIS_Y_CENTER;
    float wxOffset = xRad-THIS_X_CENTER;
    float wdOffset = dRad-THIS_D_CENTER;

    VAL_HANDLE hs = GET_NODE_VAL_HANDLE(ZN(iOutput));
    int nfSize = VAL_HANDLE_F_SIZE(hs);

    // am i doing this cast to int correctly? or should it be a round instead?
    for (int d = d1; d <= d2; d++) {
        int k = (int) ((wdOffset+NODE_D_CENTER(ZN(iOutput),d))/dSpace);
    for (int x = x1; x <= x2; x++) {
        int j = (int) ((wxOffset+NODE_X_CENTER(ZN(iOutput),x))/xSpace);
    for (int y = y1; y <= y2; y++) {
        int i = (int) ((wyOffset+NODE_Y_CENTER(ZN(iOutput),y))/ySpace);
        for (int nf = 0; nf < nfSize; nf++) {

            float n = READ_VAL_HANDLE(hs, nf, y, x, d, THIS_N);
            float w = READ_WEIGHT_VAL_HANDLE(hw, f, i, j, k, nf);

            s += n * w;

        }
    }
    }
    }
}

VAL_HANDLE ame = GET_NODE_VAL_HANDLE(ZME);    
s *= DSigmoid(READ_VAL_HANDLE(ame, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N));

WRITE_VAL(s);
