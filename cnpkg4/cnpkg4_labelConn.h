// Block size is usually [16 N] where N is found by experiment.  See CNS manual.
// N might need to be reduced for cards below GTX 285/295.
#BLOCKSIZE 16 16

int y1 = THIS_Y + READ_INDEX_VAL(ZIN, 0, ITER_NO, THIS_N) + OFFSET(0) + READ_NHOOD1(THIS_F,0);
int x1 = THIS_X + READ_INDEX_VAL(ZIN, 1, ITER_NO, THIS_N) + OFFSET(1) + READ_NHOOD1(THIS_F,1);
int d1 = THIS_D + READ_INDEX_VAL(ZIN, 2, ITER_NO, THIS_N) + OFFSET(2) + READ_NHOOD1(THIS_F,2);
int y2 = THIS_Y + READ_INDEX_VAL(ZIN, 0, ITER_NO, THIS_N) + OFFSET(0) + READ_NHOOD2(THIS_F,0);
int x2 = THIS_X + READ_INDEX_VAL(ZIN, 1, ITER_NO, THIS_N) + OFFSET(1) + READ_NHOOD2(THIS_F,1);
int d2 = THIS_D + READ_INDEX_VAL(ZIN, 2, ITER_NO, THIS_N) + OFFSET(2) + READ_NHOOD2(THIS_F,2);
int labelset = READ_INDEX_VAL(ZIN, 4, ITER_NO, THIS_N);

float l1 = READ_LABELBLOCK(labelset, 0, y1, x1, d1);
float l2 = READ_LABELBLOCK(labelset, 0, y2, x2, d2);
float l = (l1==l2) && (l1 != 0) && (l2 != 0);

float m1 = READ_MASKBLOCK(labelset, 0, y1, x1, d1);
float m2 = READ_MASKBLOCK(labelset, 0, y2, x2, d2);
int isValid = (y1>=0) && (y2>=0) && (x1>=0) && (x2>=0) && (d1>=0) && (d2>=0);
float m = m1*m2*isValid;

WRITE_VAL(l);
WRITE_MASK(m);
