int yoff1 = READ_INDEX_VAL(ZIN, 0, ITER_NO, THIS_N) + OFFSET(0) + READ_NHOOD1(THIS_F,0);
int xoff1 = READ_INDEX_VAL(ZIN, 1, ITER_NO, THIS_N) + OFFSET(1) + READ_NHOOD1(THIS_F,1);
int doff1 = READ_INDEX_VAL(ZIN, 2, ITER_NO, THIS_N) + OFFSET(2) + READ_NHOOD1(THIS_F,2);
int yoff2 = READ_INDEX_VAL(ZIN, 0, ITER_NO, THIS_N) + OFFSET(0) + READ_NHOOD2(THIS_F,0);
int xoff2 = READ_INDEX_VAL(ZIN, 1, ITER_NO, THIS_N) + OFFSET(1) + READ_NHOOD2(THIS_F,1);
int doff2 = READ_INDEX_VAL(ZIN, 2, ITER_NO, THIS_N) + OFFSET(2) + READ_NHOOD2(THIS_F,2);
int labelset = READ_INDEX_VAL(ZIN, 4, ITER_NO, THIS_N);

int l1 = READ_LABELBLOCK(labelset, 0, THIS_Y + yoff1, THIS_X + xoff1, THIS_D + doff1);
int l2 = READ_LABELBLOCK(labelset, 0, THIS_Y + yoff2, THIS_X + xoff2, THIS_D + doff2);
float l = (l1==l2) && (l1 != 0) && (l2 != 0);

float m1 = READ_MASKBLOCK(labelset, 0, THIS_Y + yoff1, THIS_X + xoff1, THIS_D + doff1);
float m2 = READ_MASKBLOCK(labelset, 0, THIS_Y + yoff2, THIS_X + xoff2, THIS_D + doff2);
float m = m1*m2;

WRITE_VAL(l);
WRITE_MASK(m);
