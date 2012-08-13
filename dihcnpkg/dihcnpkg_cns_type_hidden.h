#PART phase2


    

if (PHASE_NO == 2) {
    int f = THIS_F;
    WEIGHT_VAL_HANDLE hw = GET_WEIGHT_VAL_HANDLE(ZNW);
    int sSize = WEIGHT_VAL_HANDLE_Y_SIZE(hw);
    int dSize = WEIGHT_VAL_HANDLE_D_SIZE(hw);

    int vy1, vy2, y1, dummy;
    int vx1, vx2, x1;
    int vd1, vd2, d1;
    GET_LAYER_Y_RF_NEAR(ZN, sSize, vy1, vy2, y1, dummy);
    GET_LAYER_X_RF_NEAR(ZN, sSize, vx1, vx2, x1, dummy);
    GET_LAYER_D_RF_NEAR(ZN, dSize, vd1, vd2, d1, dummy);
    int i1 = vy1 - y1;
    int j1 = vx1 - x1;
    int k1 = vd1 - d1;

    SENS_HANDLE hs = GET_COMPUTED_SENS_HANDLE(ZN);
    int nfSize = SENS_HANDLE_F_SIZE(hs);
    float s = 0.0f;

    for (int nf = 0; nf < nfSize; nf++) {
        for (int d = vd1, k = k1; d <= vd2; d++, k++) {
        for (int x = vx1, j = j1; x <= vx2; x++, j++) {
        for (int y = vy1, i = i1; y <= vy2; y++, i++) {

            float n = READ_SENS_HANDLE(hs, nf, y, x, d);
            float w = READ_WEIGHT_VAL_HANDLE(hw, f, i, j, k, nf);

            s += n * w;

        }
        }
        }
    }

    s *= DSigmoid(READ_VAL);

    WRITE_PRESENS(s);
} else {
    
    float s = READ_PRESENS;
    
    int f = THIS_F;
        
    if (f <= 6) { //fixlater, specific
        s = DSigmoid(s);
    } else {
        PREACT_HANDLE ah = GET_COMPUTED_PREACT_HANDLE(THIS_Z);
        PRESENS_HANDLE ph = GET_HIDDEN_PRESENS_HANDLE(THIS_Z);
        
        float v = READ_PREACT;
        int x = THIS_X;
        int y = THIS_Y;
        int d = THIS_D; 
        
        if (f % 2 == 0) { //fixlater, specific
            f--;
        } else {
            f++;
        }
        
        float ov = READ_PREACT_HANDLE(ah, f, y, x, d);
        float os = READ_PRESENS_HANDLE(ph, f, y, x, d);
        
        float r = sqrt(pow(v,2)+pow(ov,2));
        float divr = 1.0f/(1.0f+r);
        s = s*divr*(1.0f-divr/r*pow(v,2)) - os*ov*v*pow(divr,2)/r; 
    }
    
    WRITE_SENS(s);
}
        
    