int yoff = READ_INDEX_VAL(ZIN, 0, ITER_NO);
int xoff = READ_INDEX_VAL(ZIN, 1, ITER_NO);
int doff = READ_INDEX_VAL(ZIN, 2, ITER_NO);

float v = READ_INPUT(THIS_Y + yoff, THIS_X + xoff, THIS_D + doff);


WRITE_VAL(v);
