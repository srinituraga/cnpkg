// Block size is usually [16 N] where N is found by experiment.  See CNS manual.
// N might need to be reduced for cards below GTX 285/295.
#BLOCKSIZE 16 16

float o = READ_NODE_VAL(ZME, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);
float l = READ_NODE_VAL(ZL, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);
float m = READ_LABEL_MASK(ZL, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);

// binary square-square margin loss
float margin = PARAM;
l = l>BINARYTHRESHOLD;
float pos = fmaxf(0.0f,0.5f-o+margin);
float neg = -fmaxf(0.0f,o-0.5f+margin);
float loss = m * 0.5*(l*pos*pos + (1-l)*neg*neg);
float dloss = m * (l*pos + (1-l)*neg) * DSigmoid(o);
float classerr = m * ((bool)l ^ (o > 0.5));

WRITE_VAL(dloss);
WRITE_LOSS(loss);
WRITE_CLASSERR(classerr);
