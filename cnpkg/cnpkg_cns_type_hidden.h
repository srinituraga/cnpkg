#PART phase2

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

	int f = THIS_F;
	float s = 0.0f;

	for (int nf = 0; nf < nfSize; nf++) {
		for (int d = vd1, k = k1; d <= vd2; d++, k++) {
		for (int x = vx1, j = j1; x <= vx2; x++, j++) {
		for (int y = vy1, i = i1; y <= vy2; y++, i++) {

			float n = READ_SENS_HANDLE(hs, nf, y, x, d, THIS_N);
			float w = READ_WEIGHT_VAL_HANDLE(hw, f, i, j, k, nf);

			s += n * w;

		}
		}
		}
	}

	s *= DSigmoid(READ_VAL);

	WRITE_SENS(s);
