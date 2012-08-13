INLINE float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

INLINE float DSigmoid(float y) {
    return y * (1.0f - y);
}

