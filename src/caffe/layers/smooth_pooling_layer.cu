#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/smooth_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "iostream"
#include "stdio.h"

namespace caffe {


  template <typename Dtype>
    __global__ void SmoothPoolForward(const int nthreads, const Dtype* bottom_data, const int num, const int channels, const int dim, int* index_data, Dtype* value_data, const bool unique_smooth, const bool has_smooth_blobs, const Dtype z, const Dtype* smooth_data, Dtype* weight, Dtype* w_norm_data, Dtype* top_data) {
      CUDA_KERNEL_LOOP(index, nthreads) {
	const int n = index / channels;
	const int c = index % channels;
	int *U, *G, *L, *ind_tmp;
	U = index_data + 3 * dim * index;
	G = U + dim;
	L = G + dim;
	const Dtype* cur_bottom = bottom_data + dim * index;
	Dtype* v_tmp = value_data + dim * index;
	Dtype* w = weight + dim * index;
	Dtype* w_norm = w_norm_data + index;
	Dtype* o = top_data + index;
	double theta,  w_tmp;
	Dtype mu;
	if (has_smooth_blobs) {
	  if (unique_smooth) {
	    mu = smooth_data[0];
	  } else {
	    mu = smooth_data[c];
	  }
	} else {
	  if (unique_smooth) {
	    mu = smooth_data[n];
	  } else {
	    mu = smooth_data[index];
	  }
	}

	for (int i = 0; i < dim; i++) {
	  v_tmp[i] = Dtype(1) / (mu + Dtype(FLT_MIN)) * cur_bottom[i];
	  U[i] = i;
	}
	double s = 0, ds = 0, ro = 0, dro = 0;
	int n_U, n_G, n_L; 
	n_U = dim;
	while (n_U > Dtype(0)) {
	  int k = n_U-1;
	  n_G = 0; n_L =0;
	  ds = 0;
	  for(int i = 0; i < n_U; i++) {
	    if (v_tmp[U[i]] >= v_tmp[U[k]]) {
	      G[n_G++] = U[i];
	      ds += double(v_tmp[U[i]]);
	    } else {
	      L[n_L++] = U[i];
	    }
	  }
	  dro = double(n_G);

	  if ((s+ds) -(ro + dro) * double(v_tmp[U[k]]) < z) {
	    s += ds; ro += dro;
	    ind_tmp = U;
	    U = L;
	    n_U = n_L;
	    L = ind_tmp;
	  } else {
	    ind_tmp = U;
	    U = G;
	    n_U = n_G -1;
	    G = ind_tmp;
	  }
	}
	theta = (s-double(z)) / (ro + DBL_MIN);

	o[0] = Dtype(0);
	w_norm[0] = 0;
	for (int i = 0; i < dim; i++) {
	  w_tmp = double(v_tmp[i]) - theta;
	  w_tmp = w_tmp > 0 ? w_tmp : double(0);
	  w[i] = Dtype(w_tmp);
	  w_norm[0] += w[i] * w[i];
	  o[0] += cur_bottom[i] * Dtype(w[i]);
	}
	o[0] -= Dtype(0.5) * mu * w_norm[0];
      }
    }



  template <typename Dtype>
    void SmoothPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      int count = top[0]->count();
      const Dtype* smooth_data = smooth_->gpu_data();
      Dtype* weight_data =  weight_.mutable_gpu_data();
      Dtype* w_norm_data = w_norm_.mutable_gpu_data();
      Blob<int> index_set(3*num_, channels_, height_, width_);
      int* index_data = index_set.mutable_gpu_data();
      Blob<Dtype> value(num_, channels_, height_, width_);
      Dtype* value_data = value.mutable_gpu_data();
      SmoothPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, num_, channels_, dim_, 
	  index_data, value_data, unique_smooth_, has_smooth_blobs_, z_, smooth_data, weight_data, w_norm_data, top_data);
      CUDA_POST_KERNEL_CHECK;
    }


  //
  //
  //template <typename Dtype>
  //__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff, const int num, const int channels, const int height,
  //                                const int width, const int pooled_height, const int pooled_width,
  //                                const int kernel_h, const int kernel_w, const int stride_h,
  //                                const int stride_w, const int pad_h, const int pad_w,
  //                                Dtype* const bottom_diff) {
  //                                  CUDA_KERNEL_LOOP(index, nthreads) {
  //                                    // find out the local index
  //                                    // find out the local offset
  //                                    const int w = index % width + pad_w;
  //                                    const int h = (index / width) % height + pad_h;
  //                                    const int c = (index / width / height) % channels;
  //                                    const int n = index / width / height / channels;
  //                                    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
  //                                    const int phend = min(h / stride_h + 1, pooled_height);
  //                                    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
  //                                    const int pwend = min(w / stride_w + 1, pooled_width);
  //                                    Dtype gradient = 0;
  //                                    const Dtype* const top_diff_slice =
  //                                    top_diff + (n * channels + c) * pooled_height * pooled_width;
  //                                    for (int ph = phstart; ph < phend; ++ph) {
  //                                      for (int pw = pwstart; pw < pwend; ++pw) {
  //                                        // figure out the pooling size
  //                                        int hstart = ph * stride_h - pad_h;
  //                                        int wstart = pw * stride_w - pad_w;
  //                                        int hend = min(hstart + kernel_h, height + pad_h);
  //                                        int wend = min(wstart + kernel_w, width + pad_w);
  //                                        int pool_size = (hend - hstart) * (wend - wstart);
  //                                        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
  //                                      }
  //                                    }
  //                                    bottom_diff[index] = gradient;
  //                                  }
  //                                }

  template <typename Dtype>
    __global__ void SmoothPoolBackward(const int nthreads, const Dtype* top_diff, const int num, const int channels, const Dtype* w_norm_data, Dtype* smooth_diff) {
      CUDA_KERNEL_LOOP(index, nthreads) {
	const Dtype* cur_top_diff = top_diff + index;
	const Dtype* cur_w_norm = w_norm_data + index;

	for (int i = 0; i < num; i++) {
	  smooth_diff[index] += cur_top_diff[i*channels] * cur_w_norm[i*channels];
	}
	smooth_diff[index] *= -Dtype(0.5); 
      }
    }

  template <typename Dtype>
    __global__ void caffe_gpu_hadamard_product(const int nthreads, const Dtype alpha, const Dtype* a, const Dtype* b, Dtype* c) {
      CUDA_KERNEL_LOOP(index, nthreads) {
	c[index] = alpha * a[index] * b[index];
      }
    }



  template <typename Dtype>
    void SmoothPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      LOG(INFO) << "start backward_gpu";
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* weight_data = weight_.gpu_data();
      const Dtype* w_norm_data = w_norm_.gpu_data();
      if (propagate_down[0]) {
	//Gradient with respect to bottom [0]
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* top_cpu_diff = top[0]->cpu_diff();
	caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
	for (int n = 0; n < num_; n++) {
	  for (int c = 0; c < channels_; c++) {
	    const Dtype* cur_weight_data = weight_data + weight_.offset(n, c);
	    Dtype* cur_bottom_diff = bottom_diff + bottom[0]->offset(n, c);
	    const Dtype* cur_top_diff = top_cpu_diff + top[0]->offset(n, c); 
	    caffe_gpu_axpy(dim_, cur_top_diff[0], cur_weight_data,  cur_bottom_diff);
	  }
	}
      }
      if (!has_smooth_blobs_ && propagate_down[1]) {
	Dtype* smooth_diff = smooth_->mutable_gpu_diff();
	Dtype* smooth_cpu_diff = smooth_->mutable_cpu_diff();
	caffe_gpu_set(smooth_->count(), Dtype(0), smooth_diff);
	if (unique_smooth_) {
	  for (int n = 0; n < num_; n++) {
	    Dtype* cur_smooth_diff = smooth_cpu_diff + n;
	    const Dtype* cur_w_norm = w_norm_data + w_norm_.offset(n);
	    const Dtype* cur_top_diff = top_diff + top[0]->offset(n);
	    caffe_gpu_dot(channels_, cur_w_norm, cur_top_diff, cur_smooth_diff);
	    cur_smooth_diff[0] *= -Dtype(0.5);
	  }
	} else {
	  int count = top[0]->count();
	  caffe_gpu_hadamard_product<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, -Dtype(0.5), top_diff, w_norm_data, smooth_diff);
	}
      } else if (has_smooth_blobs_ && this->param_propagate_down_[0]) {
	// Gradient with respect to smooth_ param
	Dtype* smooth_diff = smooth_->mutable_gpu_diff();
	Dtype* smooth_cpu_diff = smooth_->mutable_cpu_diff();
	//caffe_gpu_set(smooth_->count(), Dtype(0), smooth_diff);
	if (unique_smooth_) {
	  int count = top[0]->count();
	  caffe_gpu_dot(count, w_norm_data, top_diff, smooth_cpu_diff);
	  smooth_cpu_diff[0] *= -Dtype(0.5);
	  LOG(INFO) << "gpu Smooth diff: " << smooth_cpu_diff[0];
	} else {
	  SmoothPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(channels_), CAFFE_CUDA_NUM_THREADS>>>(channels_, top_diff, num_, channels_,  w_norm_data, smooth_diff);
	  for (int i = 0; i < channels_; i++) {
	  LOG(INFO) << "gpu Smooth diff: " << smooth_cpu_diff[i];
         }
 }
      }
      CUDA_POST_KERNEL_CHECK;
    }

  //  if (!propagate_down[0]) {
  //    return;
  //  }
  //  const Dtype* top_diff = top[0]->gpu_diff();
  //  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  //  const int count = bottom[0]->count();
  //  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  //  // We'll output the mask to top[1] if it's of size >1.
  //  const bool use_top_mask = top.size() > 1;
  //  const int* mask = NULL;
  //  const Dtype* top_mask = NULL;
  //  switch (this->layer_param_.pooling_param().pool()) {
  //    case PoolingParameter_PoolMethod_MAX:
  //      if (use_top_mask) {
  //        top_mask = top[1]->gpu_data();
  //      } else {
  //        mask = max_idx_.gpu_data();
  //      }
  //      // NOLINT_NEXT_LINE(whitespace/operators)
  //      MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //          count, top_diff, mask, top_mask, top[0]->num(), channels_,
  //          height_, width_, pooled_height_, pooled_width_,
  //          kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
  //          bottom_diff);
  //      break;
  //    case PoolingParameter_PoolMethod_AVE:
  //      // NOLINT_NEXT_LINE(whitespace/operators)
  //      AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //          count, top_diff, top[0]->num(), channels_,
  //          height_, width_, pooled_height_, pooled_width_, kernel_h_,
  //          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
  //      break;
  //    case PoolingParameter_PoolMethod_STOCHASTIC:
  //      // NOLINT_NEXT_LINE(whitespace/operators)
  //      StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //          count, rand_idx_.gpu_data(), top_diff,
  //          top[0]->num(), channels_, height_, width_, pooled_height_,
  //          pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
  //          bottom_diff);
  //      break;
  //    default:
  //      LOG(FATAL) << "Unknown pooling method.";
  //  }
  //  CUDA_POST_KERNEL_CHECK;


  INSTANTIATE_LAYER_GPU_FUNCS(SmoothPoolingLayer);


}  // namespace caffe
