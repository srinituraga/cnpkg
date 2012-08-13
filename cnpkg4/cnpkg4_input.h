// Block size is usually [16 N] where N is found by experiment.  See CNS manual.
// N might need to be reduced for cards below GTX 285/295.
#BLOCKSIZE 16 16

int yoff = READ_INDEX_VAL(ZIN(0), 0, ITER_NO, THIS_N);
int xoff = READ_INDEX_VAL(ZIN(0), 1, ITER_NO, THIS_N);
int doff = READ_INDEX_VAL(ZIN(0), 2, ITER_NO, THIS_N);
int set = READ_INDEX_VAL(ZIN(0), 3, ITER_NO, THIS_N);

float v = READ_INPUTBLOCK(set, THIS_F, THIS_Y + yoff, THIS_X + xoff, THIS_D + doff);

WRITE_VAL(v);
