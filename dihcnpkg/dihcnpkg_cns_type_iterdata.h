//#TEMPLATE

if (THIS_I == ITER_NO) {
    LAYER_VAL_HANDLE vh = GET_LAYER_VAL_HANDLE(ZOUT);
    OUTPUT_LABEL_HANDLE lh = GET_OUTPUT_LABEL_HANDLE(ZOUT);
    int sSize = LAYER_VAL_HANDLE_Y_SIZE(vh);
    int dSize = LAYER_VAL_HANDLE_D_SIZE(vh);
    int fSize = LAYER_VAL_HANDLE_F_SIZE(vh);

    float err = 0;
    
    float numel = sSize*sSize*dSize*fSize;
    
    for (int f = 0; f < fSize; f++) {
        for (int d = 0; d < dSize; d++) {
            for (int x = 0; x < sSize; x++) {
               for (int y = 0; y < sSize; y++) {
                    float t = READ_OUTPUT_LABEL_HANDLE(lh, f, y, x, d);
                    float c = READ_LAYER_VAL_HANDLE(vh, f, y, x, d);
                   
                    err += LossFunction(t,c)/numel;
                    //doesn't account for masking

                }
            }
        }
    }
    
    WRITE_ERR(err);
}
        
