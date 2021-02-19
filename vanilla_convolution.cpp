#include <iostream>
#include <random>
#include <stdio.h>
#include <sys/time.h>

int main() {
    constexpr int b = 1;
    constexpr int ic = 3;
    constexpr int h = 5;
    constexpr int w = 5;
  
    constexpr int ph = 0;
    constexpr int pw = 0;
    constexpr int sh = 1;
    constexpr int sw = 1;
  
    constexpr int output_channel = 3;
    constepxr int fh = 3;
    constexpr int fw = 3;
  
    unsigned int random_seed = 123;
    float* input = new float[b * ic * h * w];
    float* weights = new float[output_channel * ic * fh * fw];
    
    for (int i = 0; i < b * ic * h * w; i++) input[i] = (float)i;
    for (int i = 0; i < output_channel * ic * fh * fw; i++) weights[i] = (float)i;
    
    cout << "==================input=================" << endl;
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < ic; j++) {
            for (int k = 0; k < h; k++) {
                for (int l = 0; l < w; l++) {
                    printf("%.4f ", input[i * ic * h * w + j * h * w + k * w + l];
                }
            }
        }
    }
    
    // do vanilla convolution
    int output_h = (h + 2 * ph - fh ) / sh + 1;
    int output_w = (w + 2 * pw - fw ) / sw + 1;
    
    // output_height & width with no stride
    int output_h_no_stride = 1 + (h + 2 * ph - fh);
    int output_w_no_stride = 1 + (w + 2 * pw - fw);
    
    float* output = new float[output_channel * output_h * output_w * b];
    for (int oc = 0; oc < output_channel; oc++) {
        for (int oh = 0; oh < output_h_no_stride; oh += sh) {
            for (int ow = 0; ow < output_w_no_stride; ow += sw) {
                float partial_sum = 0.0f;
                for (int c = 0; c < ic; c++) {
                    for (int kh = 0; kh < fh; kh++) {
                        for (int kw = 0; kw < fw; kw++) {
                            // boundary check
                            if ((ow + fw) > w + pw || (oh + fh) > h + ph) continue;
                            // TODO(jyeon.kim) : consider the padding check
                            int input_index = c * h * w + (oh - ph + kh) * w + (ow - pw + kw);
                            int kernel_index = oc * ic * fh * fw + c * fh * fw + kh * fw + kw;
                            partial_sum += input[input_index] * weights[kernel_index];
                        }
                    }
                }
                int output_index = oc * output_h * output_w + (oh / sh) * output_w + (ow / sw);
                output[output_index] = partial_sum;
            }
        }
    }
    
    cout << "====================output=======================" << endl;
    for (int i = 0; i < output_channel; i++) {
        for (int j = 0; j < output_h; j++) {
            for (int k = 0; k < output_w; k++) {
                printf("%.4f ", output[i * output_h * output_w + j * output_w + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}
