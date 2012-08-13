// Block size is usually [16 N] where N is found by experiment.  See CNS manual.
// N might need to be reduced for cards below GTX 285/295.
#BLOCKSIZE 16 16

float o = READ_NODE_VAL(ZME, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);
float l = READ_NODE_VAL(ZL, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);
float m = READ_LABEL_MASK(ZL, THIS_F, THIS_Y, THIS_X, THIS_D, THIS_N);

float dloss = (l-o) * DSigmoid(o) * m;
float loss = (l-o)*(l-o) * m;
//float classerr = ((l>BINARYTHRESHOLD)^(o>BINARYTHRESHOLD)) * m;
float classerr = ((l>BINARYTHRESHOLD)^(o>.5)) * m;

WRITE_VAL(dloss);
WRITE_LOSS(loss);
WRITE_CLASSERR(classerr);
