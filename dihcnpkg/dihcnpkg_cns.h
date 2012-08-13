INLINE float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

INLINE float DSigmoid(float y) {
    return y * (1.0f - y);
}

INLINE float Hillfxn(float x, float y) {
    return x / (1.0f + sqrtf(x*x+y*y));
}


//_INLINE float LossFunction(float x, float y) {
//    return (x - y)^2;
//}

INLINE float LossFunction(float x, float y) {
    return (x - y)*(x - y);
}

INLINE float DLossFunction(float x, float y) {
    return -(x - y);
}

