#TEMPLATE

if (PHASE_NO == 0) {

    WEIGHT_VAL_HANDLE hw = GET_WEIGHT_VAL_HANDLE(ZW);
    int sSize = WEIGHT_VAL_HANDLE_Y_SIZE(hw);
    int dSize = WEIGHT_VAL_HANDLE_D_SIZE(hw);

    int y1, x1, d1, dummy;
    GET_LAYER_Y_RF_NEAR(ZP, sSize, y1, dummy);
    GET_LAYER_X_RF_NEAR(ZP, sSize, x1, dummy);
    GET_LAYER_D_RF_NEAR(ZP, dSize, d1, dummy);

    VAL_HANDLE hv = GET_LAYER_VAL_HANDLE(ZP);
    int fSize = VAL_HANDLE_F_SIZE(hv);

	int nf = THIS_F;

	float v = READ_BIAS_VAL(ZB, nf, 0);

	for (int f = 0; f < fSize; f++) {
		for (int k = dSize - 1, d = d1; k >= 0; k--, d++) {
		for (int j = sSize - 1, x = x1; j >= 0; j--, x++) {
		for (int i = sSize - 1, y = y1; i >= 0; i--, y++) {

			float p = READ_VAL_HANDLE(hv, f, y, x, d, THIS_N);
			float w = READ_WEIGHT_VAL_HANDLE(hw, f, i, j, k, nf);

			v += p * w;

		}
		}
		}
	}
	v = Sigmoid(v);
	WRITE_VAL(v);

} else {

    #PART phase2

}
