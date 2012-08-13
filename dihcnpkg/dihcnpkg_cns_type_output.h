#PART phase2

    float v = READ_VAL;

    int yoff = READ_INDEX_VAL(ZIN, 0, ITER_NO);
    int xoff = READ_INDEX_VAL(ZIN, 1, ITER_NO);
    int doff = READ_INDEX_VAL(ZIN, 2, ITER_NO);
    
    int yAdditionalOffset = READ_INDEX_ADDITIONALOFFSET(ZIN, 0, ITER_NO);
    int xAdditionalOffset = READ_INDEX_ADDITIONALOFFSET(ZIN, 1, ITER_NO);
    int dAdditionalOffset = READ_INDEX_ADDITIONALOFFSET(ZIN, 2, ITER_NO);

    float c = READ_LABELBLOCK(THIS_Y + yoff + yAdditionalOffset, THIS_X + xoff + xAdditionalOffset, THIS_D + doff + dAdditionalOffset);
//    float mask = READ_LABELMASKBLOCK(THIS_Y + yoff + yAdditionalOffset, THIS_X + xoff + xAdditionalOffset, THIS_D + doff + dAdditionalOffset);
    
    //float c = READ_LABEL;
    //float mask = READ_LABELM;
    
    WRITE_LABEL(c);
//    WRITE_LABELM(mask);
    
    //float s = DSigmoid(v) * (c - v) * mask;
    float s = DLossFunction(c,v);
    WRITE_SENS(s);
    
    
    /*PRINT("yoff, THIS_Y, yAdditionalOffset =%d, %d, %d", yoff, THIS_Y, yAdditionalOffset);
    PRINT("xoff, THIS_X, xAdditionalOffset =%d, %d, %d", xoff, THIS_X, xAdditionalOffset);
    PRINT("doff, THIS_D, dAdditionalOffset =%d, %d, %d", doff, THIS_D, dAdditionalOffset);

    PRINT("v=%e, DSigmoid(v)=%e, c=%e, (c-v)=%e, mask=%e, s=%e", (double)v, (double)DSigmoid(v), (double)c, (double)(c-v), (double)mask, (double)s);*/
        
