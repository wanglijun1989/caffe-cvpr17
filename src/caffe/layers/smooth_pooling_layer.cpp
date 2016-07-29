#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/smooth_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
  template <typename Dtype>
  void project_simplex(const Dtype* v, int n, Dtype mu, Dtype z, Dtype* w) {
    double theta, w_tmp;
    int *U, *G, *L;
    U = new int [n];
    G = new int [n];
    L = new int [n];
    Dtype* v_tmp = new Dtype[n];
    for (int i = 0; i < n; i++) {
      v_tmp[i] = Dtype(1)/mu * v[i];
      U[i] = i;
    }
    double s = 0, ds = 0, ro = 0, dro = 0;
    int n_U, n_G, n_L; 
    n_U = n;
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
        delete [] U;
        U = L;
        n_U = n_L;
        L = new int [n_U];
      } else {
        delete [] U;
        U = G;
        n_U = n_G -1;
        G = new int [n_U];
      }
    }
    delete [] U;
    delete [] G;
    delete [] L;
    theta = (s-z) / ro;
    for(int i = 0; i < n; i++) {
      w_tmp = double(v_tmp[i]) - theta;
      w_tmp = w_tmp > 0 ? w_tmp : Dtype(0);
      w[i] = double(w_tmp);
    } 
    //delete [] v_tmp;
  }
template
void project_simplex<float>(const float* v, int n, float mu, float z, float* w);
template 
void project_simplex<double>(const double* v, int n, double mu, double z, double* w);

  template <typename Dtype>
  void SmoothPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LayerParameter param = this->layer_param_;
    SmoothPoolingParameter pool_param = param.smooth_pooling_param();
    z_ = pool_param.z();
    unique_smooth_ = pool_param.unique_smooth(); 
    has_smooth_blobs_ = pool_param.has_smooth_blobs();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    dim_ = height_ * width_;
    CHECK((bottom.size() > 1 || has_smooth_blobs_) && !(bottom.size() > 1 && has_smooth_blobs_)) << "Smooth parameters should be provided by eitehr bottom blobs or layer parameter blobs but not both.";

    if (bottom.size() > 1) {
      // bottom blob provides smooth parameters
      if (unique_smooth_) {
        CHECK(bottom[1]->width() == 1 && bottom[1]->height() == 1 && bottom[1]->channels() == 1) << "The size of smooth parameters is wrong.";
      } else {
        CHECK(bottom[1]->width() == 1 && bottom[1]->height() == 1 && bottom[1]->channels() == channels_) << "The size of smooth parameters is wrong.";
      }
      smooth_.reset(bottom[1]);
    } else {
      this->blobs_.resize(1);
      if (unique_smooth_) {
        this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1)); 
      } else {
        this->blobs_[0].reset(new Blob<Dtype>(1, channels_, 1, 1)); 
      }
      shared_ptr<Filler<Dtype> > smooth_filler(GetFiller<Dtype>(pool_param.smooth_filler()));
      smooth_filler->Fill(this->blobs_[0].get());
      smooth_ = this->blobs_[0];
      this->param_propagate_down_.resize(1, true);
    }
    weight_.Reshape(num_, channels_, height_, width_);
    LOG(INFO) << "Setup done";
  }

  template <typename Dtype>
  void SmoothPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";
    CHECK_EQ(channels_, bottom[0]->channels()) << "Channel number for SmoothPooling layer should be fixed after layer setup.";
    num_ = bottom[0]->num();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    dim_ = height_ * width_;
    weight_.Reshape(num_ , channels_, height_, width_);
    if (!has_smooth_blobs_) {
      // smooth blobs is provided by bottom[1]
      smooth_.reset(bottom[1]);
    }
    top[0]->Reshape(num_, channels_, 1, 1);
    LOG(INFO) << "Reshape done";
  }

  template <typename Dtype>
  void SmoothPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    LOG(INFO) << "0" ;
    Dtype* top_data = top[0]->mutable_cpu_data();
    LOG(INFO) << "1";
    Dtype* weight_data = weight_.mutable_cpu_data();
    LOG(INFO) << "2";
    const Dtype* smooth_data = smooth_->cpu_data();
    LOG(INFO) << "3";
    // First compute weight_ for each num and channel
    // Second weighted average for each channel
    for (int n = 0; n < num_; n++) {
      for (int c = 0; c < channels_; c++) {
        const Dtype* cur_bottom = bottom_data + bottom[0]->offset(n, c);
        Dtype* cur_top = top_data + top[0]->offset(n, c);
        Dtype* cur_weight = weight_data + weight_.offset(n, c);
        Dtype cur_smooth;
        if (has_smooth_blobs_) {
          cur_smooth = unique_smooth_ ? smooth_data[0] : smooth_data[c];
        } else {
          cur_smooth = unique_smooth_ ? smooth_data[n] : smooth_data[n * channels_ + c];
        }
        project_simplex(cur_bottom, dim_, cur_smooth, z_, cur_weight);
        *cur_top = caffe_cpu_dot(dim_, cur_bottom, cur_weight);
        // TODO: compute weighted average in a batch mode via matrix matrix multiplication
      }
    }
  }

template <typename Dtype>
void SmoothPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weight_data = weight_.mutable_cpu_data();
  if (propagate_down[0]) {
    // Gradient with respect to bottom [0]
    // set bottom[0] diff to zeors
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    for (int n = 0; n < num_; n++) {
      for (int c = 0; c < channels_; c++) {
        const Dtype* cur_weight_data = weight_data + weight_.offset(n, c);
        Dtype* cur_bottom_diff = bottom_diff + bottom[0]->offset(n, c);
        const Dtype* cur_top_diff = top_diff + top[0]->offset(n, c); 
        caffe_axpy(dim_, *cur_top_diff, cur_weight_data,  cur_bottom_diff);
      }
    }
  }
  if ( (!has_smooth_blobs_ && propagate_down[1]) || (has_smooth_blobs_ && this->param_propagate_down_[0]) ) {
    // Gradient with respect to bottom[1]
    // Gradient with respect to smooth_
    Dtype* smooth_diff = smooth_->mutable_cpu_diff();
    caffe_set(smooth_->count(), Dtype(0), smooth_diff);
    if (unique_smooth_) {
      for (int n = 0; n < num_; n++) {
        Dtype* cur_smooth_diff = smooth_diff + n;
        for (int c = 0; c < channels_; c++) {
          const Dtype* cur_top_diff = top_diff + top[0]->offset(n, c); 
          const Dtype* cur_weight_data = weight_data + weight_.offset(n, c);
          *cur_smooth_diff += -Dtype(0.5) * caffe_cpu_dot(dim_, cur_weight_data, cur_weight_data) * cur_top_diff[0];
        }
      }
    } else {
      for (int n = 0; n < num_; n++) {
        for (int c = 0; c < channels_; c++) {
          Dtype* cur_smooth_diff = smooth_diff + smooth_->offset(n, c);
          const Dtype* cur_top_diff = top_diff + top[0]->offset(n, c); 
          const Dtype* cur_weight_data = weight_data + weight_.offset(n, c);
          *cur_smooth_diff = -Dtype(0.5) * caffe_cpu_dot(dim_, cur_weight_data, cur_weight_data) * cur_top_diff[0];
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SmoothPoolingLayer);
#endif

INSTANTIATE_CLASS(SmoothPoolingLayer);

}  // namespace caffe
