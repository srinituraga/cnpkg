#PART phase2

    float o = READ_VAL;
    float l = READ_LAYER_VAL(ZL, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);
    float m = READ_LABEL_MASK(ZL, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);

    float dloss = DLossFunction(l, o) * m;
    float loss = LossFunction(l, o) * m;
	float classerr = ((int)(l > 0.5f)) ^ ((int)(o > 0.5f));

    WRITE_SENS(dloss);
    WRITE_LOSS(loss);
	WRITE_CLASSERR(classerr);
