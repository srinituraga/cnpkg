// Block size is usually [16 N] where N is found by experiment.  See CNS manual.
// N might need to be reduced for cards below GTX 285/295.
#BLOCKSIZE 16 16

int yoff = READ_INDEX_VAL(ZIN, 0, ITER_NO, THIS_N) + OFFSET(0);
int xoff = READ_INDEX_VAL(ZIN, 1, ITER_NO, THIS_N) + OFFSET(1);
int doff = READ_INDEX_VAL(ZIN, 2, ITER_NO, THIS_N) + OFFSET(2);
int labelset = READ_INDEX_VAL(ZIN, 4, ITER_NO, THIS_N);

float l = READ_LABELBLOCK(labelset, THIS_F, THIS_Y + yoff, THIS_X + xoff, THIS_D + doff);
float m = READ_MASKBLOCK(labelset, THIS_F, THIS_Y + yoff, THIS_X + xoff, THIS_D + doff);

WRITE_VAL(l);
WRITE_MASK(m);
