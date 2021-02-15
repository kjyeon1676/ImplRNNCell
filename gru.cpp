#include <string.h>
#include <functional>
#include <math.h>
#include <memory.h>

static void matMulFunc(void* x, void* y, void* z, int xSize, int ySize, int zSize) {
    for (int i = 0; i < xSize; i++) {
        for (int j = 0; j < ySize; j++) {
            static_cast<float*>(z)[i * ySize + j] = 0.0f;
            for (int k = 0; k < zSize; k++) {
                static_cast<float*>(z)[i * ySize + j] += static_cast<float*>(x)[i * zSize + k] * static_cast<float*>(y)[k * ySize + j];
            }
        }
     }
}

static void actFunc(void* x, int size, std::string name) {
    if (strcmp(name.c_str(), "sigmoid") == 0) {
        static_cast<float*>(x)[i] = 1 / (1 + exp(-1 * (static_cast<float*>(x)[i])));
    }
    else if (strcmp(name.c_str(), "tanh") == 0) {
        static_cast<float*>(x)[i] = tanh(static_cast<float*>(x)[i]);
    }
}

static void hadamardFunc(void* x, void* y, void* z, int size) {
    for (int i = 0; i < size; i++) {
        static_cast<float*>(z)[i] = static_cast<float*>(x)[i] * static_cast<float*>(y)[i];
    }
}

static void eltwiseAdd(void* x, void* y, void* z, int size) {
    for (int i = 0; i < size; i++) {
        static_cast<float*>(z)[i] = static_cast<float*>(x)[i] + static_cast<float*>(y)[i];
    }
}

static void sliceFunc(void* x, void* y, int xSize, int ySize, int hSize, int start) {
    for (int i = 0; i < xSize; i++) {
        for (int j = 0; j < ySize; j++) {
            static_cast<float*>(x)[i * ySize + j] = static_cast<float*>(y)[i * ySize * hSize + j + start];
        }
    }
}

static void cellForward(void* src, void* prevHidden, void* inputWeights, void* hiddenWeights, void* bias, void* dst, void* hNext, void* workspace,
                        void* reserveSpace, size_t typeSize, int batch, int channel, int hidden) {
    int c = channel;
    int h = hidden;
    // first, reset gate + update gate
    // Wx(input weights) * x
    size_t mmSize = batch * h * typeSize;
    void* rg = malloc(mmSize);
    void* ig = malloc(mmSize);
    void* wh_prev = malloc(mmSize);
    void* hh_prev = malloc(mmSize);
    
    // weight split
    size_t weightSplitSize = c * h * typeSize;
    void* wxr = malloc(weightSplitSize);
    void* wxi = malloc(weightSplitSize);
    void* wxn = malloc(weightSplitSize);
    
    weightSplitSize = h * h * typeSize;
    void* whr = malloc(weightSplitSize);
    void* whi = malloc(weightSplitSize);
    void* whn = malloc(weightSplitSize);
    
    // input weight slice
    sliceFunc(wxr, inputWeights, c, h, 3, 0);
    sliceFunc(wxi, inputWeights, c, h, 3, h);
    sliceFunc(wxn, inputWeights, c, h, 3, 2 * h);
    
    // hidden Weight slice
    sliceFunc(whr, hiddenWeights, h, h, 3, 0);
    sliceFunc(whi, hiddenWeights, h, h, 3, h);
    sliceFunc(whn, hiddenWeights, h, h, 3, 2 * h);
    
    // (N * C) x (C x H) --> N x H --> input * input weight
    // (N * H) x (H * H) --> N x H --> hidden * hidden weight
    matMulFunc(src, wxr, rg, batch ,h, c);
    matMulFunc(prevHidden, whr, wh_prev, batch, h, h,);
    
    // input gate
    matMulFunc(src, wxi, ig, batch, h, c);
    matMulFunc(prevHidden, whi, hh_prev, batch, h, h);
    
    // add rg(N * H) + wh_prev(H * H) + bias(H)
    eltwiseAdd(rg, wh_prev, rg, batch * h);
    eltwiseAdd(ig, hh_prev, ig, batch * h);
    for (int i = 0; i < batch; i++) {
        eltwiseAdd(static_cast<float*>(rg) + i * h, bias, static_cast<float*>(rg) + i * h, h);
        eltwiseAdd(static_cast<float*>(ig) + i * h, static_cast<float*>(bias) + h, static_cast<float*>(ig) + i * h, h);
    }
    
    // apply sigmoid function to a reset gate and update gate
    actFunc(rg, batch * h, "sigmoid");
    actFunc(ig, batch * h, "sigmoid");
    
    void* newGateTmp = malloc(mmSize);
    void* ng = malloc(mmSize);
    void* ngh = malloc(mmSize);
    
    // x(N * C) x wi(C * H) --> N x H
    matMulFunc(src, wxn, ng, batch, h, c);
    matMulFunc(prevHidden, whn, ngh, batch, h, h);
    hadamardFunc(ngh, rg, newGateTmp, batch * h);
    
    eltwiseAdd(ng, newGateTmp, ng, batch * h);
    for (int i = 0; i < batch; i++) eltwiseAdd(static_cast<float*>(ng) + i * h, static_cast<float*>(bias) + 2 * h, static_cast<float*>(ng) + i * h, h);
    
    // ng - newGate(g)
    actFunc(ng, batch * h, "tanh");
    
    // calculation of output gate
    // reuse wh_prev, hh_prev, ngh
    void* oih = hh_prev;
    void* oin = wh_prev;
    void* otmp = ngh;
    
    // input gate and init hidden hadamard
    hadamardFunc(ig, prevHidden, oih, batch * h);
    // (1 - input gate)
    for (int i = 0; i < batch * h; i++) {
        static_cast<float*>(oin)[i] = (1.0 - static_cast<float*>(ig)[i]);
    }
    
    // (1 - input gate * new gate)
    hadamardFunc(oin, ng, otmp, batch * h);
    eltwiseAdd(otmp, oih, dst, batch * h);
    
    // h_next(h_t-1 * update gate) + (1-update gate * new gate)
    hadamardFunc(prevHidden, ig, oih, batch * h);
    eltwiseAdd(oih, otmp, hNext, batch * h);
    
    // copy from gate data to reserve space because of backward propagation.
    memcpy(static_cast<float*>(reserveSpace), rg, mmSize);
    memcpy(static_cast<float*>(reserveSpace) + batch * h, ig, mmSize);
    memcpy(static_cast<float*>(reserveSpace) + batch * 2 * h, ng, mmSize);
    
    // free
    free(ig);
    free(rg);
    free(wh_prev);
    free(hh_prev);
    free(wxr);
    free(wxi);
    free(wxn);
    free(whr);
    free(whi);
    free(whn);
    free(newGateTmp);
    free(ng);
    free(ngh);
}

    
                                                                                
