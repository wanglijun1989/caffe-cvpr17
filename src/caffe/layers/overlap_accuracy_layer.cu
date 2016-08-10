#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/overlap_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void OverlapAccuracyForwardGPU(const int nthreads, const Dtype* prediction, const Dtype* label, Dtype* intersection, Dtype* union_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(label[index] > 0.5 && prediction[index] >= 0.5) {
      intersection[index] = 1;
      union_data[index] = 1;
    } else if (label[index] >0.5 || prediction[index] >= 0.5) {
      intersection[index] = 0;
      union_data[index] = 1;
    } else {
      intersection[index] = 0;
      union_data[index] = 0;
    }
  }
}

template <typename Dtype>
void OverlapAccuracyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  Dtype* intersection_data = intersection_.mutable_gpu_data();
  Dtype* union_data = union_.mutable_gpu_data();
  const Dtype* prediction = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data(); 
  OverlapAccuracyForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, prediction, label, intersection_data, union_data);
  Dtype I, U, accuracy;
  caffe_gpu_asum(count, intersection_data, &I);
  caffe_gpu_asum(count, union_data, &U);
  accuracy = I / U ;
  top[0]->mutable_cpu_data()[0] = accuracy;
}

INSTANTIATE_LAYER_GPU_FUNCS(OverlapAccuracyLayer);

}  // namespace caffe
