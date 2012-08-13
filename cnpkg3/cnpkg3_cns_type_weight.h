if (PHASE_NO == 0) {
// Compute the gradient

	COMPUTED_SENS_HANDLE hs = GET_COMPUTED_SENS_HANDLE(ZN);
	float ySize = COMPUTED_SENS_HANDLE_Y_SIZE(hs);
	float xSize = COMPUTED_SENS_HANDLE_X_SIZE(hs);
	float dSize = COMPUTED_SENS_HANDLE_D_SIZE(hs);

	float nSize = COMPUTED_SENS_HANDLE_N_SIZE(hs);

	LAYER_VAL_HANDLE hv = GET_LAYER_VAL_HANDLE(ZP);

	float yRad = 0.5f * LAYER_Y_SPACE(ZP) * Y_SIZE;
	float xRad = 0.5f * LAYER_X_SPACE(ZP) * X_SIZE;
	float dRad = 0.5f * LAYER_D_SPACE(ZP) * D_SIZE;

	// first part computes the first 'val' index that's relevant to the first 'sens' index
	// the second part is the shift corresponding to the weight that we are interested in
	float vy1 = ((LAYER_Y_START(ZN)-yRad-LAYER_Y_START(ZP))/LAYER_Y_SPACE(ZP)) + (Y_SIZE - 1 - THIS_Y);
	float vx1 = ((LAYER_X_START(ZN)-xRad-LAYER_X_START(ZP))/LAYER_X_SPACE(ZP)) + (X_SIZE - 1 - THIS_X);
	float vd1 = ((LAYER_D_START(ZN)-dRad-LAYER_D_START(ZP))/LAYER_D_SPACE(ZP)) + (D_SIZE - 1 - THIS_D);

	// step the val index by this much
	float deltavy = LAYER_Y_SPACE(ZN)/LAYER_Y_SPACE(ZP);
	float deltavx = LAYER_X_SPACE(ZN)/LAYER_X_SPACE(ZP);
	float deltavd = LAYER_D_SPACE(ZN)/LAYER_D_SPACE(ZP);

	int f  = THIS_F;
	int nf = THIS_NF;

	float dw = 0.0f;

	// the +0.5f is so that when vy gets cast to an integer,
	// the implicit 'floor' will round to the nearest integer instead
	float sd, sx, sy;
	float vd, vx, vy;
	float norm = 0.0f;
	for (int n = 0; n < nSize; n++) {
	for (sd = 0, vd = vd1+0.5f; sd < dSize; sd++, vd+=deltavd) {
	for (sx = 0, vx = vx1+0.5f; sx < xSize; sx++, vx+=deltavx) {
	for (sy = 0, vy = vy1+0.5f; sy < ySize; sy++, vy+=deltavy) {

		float v = READ_LAYER_VAL_HANDLE(hv, f, (int) vy, (int) vx, (int) vd, n);
		float s = READ_COMPUTED_SENS_HANDLE(hs, (int) nf, (int) sy, (int) sx, (int) sd, n);

		dw += v * s;
		norm++;

	}
	}
	}
	}

	dw /= norm;
	WRITE_DVAL(dw);

} else if (PHASE_NO == 1) {
// Apply constraints and update the weights

	float yWPxSize = WPXSIZE(0);
	float xWPxSize = WPXSIZE(1);
	float dWPxSize = WPXSIZE(2);

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

	float w = READ_VAL;
	w += GLOBALETA * ETA * dw;
	WRITE_VAL(w);

}
