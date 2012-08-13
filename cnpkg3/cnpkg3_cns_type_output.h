#PART phase2

    float o = READ_VAL;
    float l = READ_LAYER_VAL(ZL, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);
    float m = READ_LABEL_MASK(ZL, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);

    float dloss = (l-o) * DSigmoid(o) * m;
    float loss = (l-o)*(l-o) * m;
	//float classerr = ((l>BINARYTHRESHOLD)^(o>BINARYTHRESHOLD)) * m;
	float classerr = ((l>BINARYTHRESHOLD)^(o>.5)) * m;

    WRITE_SENS(dloss);
    WRITE_LOSS(loss);
	WRITE_CLASSERR(classerr);
