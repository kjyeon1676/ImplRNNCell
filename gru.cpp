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
