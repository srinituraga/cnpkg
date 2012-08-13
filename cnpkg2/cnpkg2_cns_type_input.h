int yoff = READ_INDEX_VAL(ZIN, 0, ITER_NO, THIS_N);
int xoff = READ_INDEX_VAL(ZIN, 1, ITER_NO, THIS_N);
int doff = READ_INDEX_VAL(ZIN, 2, ITER_NO, THIS_N);
int set = READ_INDEX_VAL(ZIN, 3, ITER_NO, THIS_N);

float v = READ_INPUTBLOCK(set, THIS_F, THIS_Y + yoff, THIS_X + xoff, THIS_D + doff);

WRITE_VAL(v);
