#PART phase2

    float o = READ_VAL;
    float l = READ_LAYER_VAL(ZL, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);
    float m = READ_LABEL_MASK(ZL, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);

	// binary square-square margin loss
	float margin = MARGIN;
	l = l>BINARYTHRESHOLD;
	float pos = fmaxf(0.0f,0.5f-o+margin);
	float neg = -fmaxf(0.0f,o-0.5f+margin);
	float loss = m * (l*pos*pos + (1-l)*neg*neg);
	float dloss = m * (l*pos + (1-l)*neg) * DSigmoid(o);
	float classerr = m * ((bool)l ^ (o > 0.5));

    WRITE_SENS(dloss);
    WRITE_LOSS(loss);
	WRITE_CLASSERR(classerr);
