// Contains helper functions that any kernel can use.

// Activation function y = f(x).
INLINE float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Derivative dy/dx of the activation function (in terms of y).
INLINE float DSigmoid(float y) {
    return y * (1.0f - y);
}
