#./build/tools/caffe train -solver=examples/cvpr17/exp_solver.prototxt -weights=/home/lijun/Research/Code/CVPR17/model/vgg16CAM_train_iter_90000.caffemodel 2>&1 | tee examples/cvpr17/log_exp.txt 

#./build/tools/caffe train -solver=examples/cvpr17/exp_solver.prototxt -gpu=0,1 2>&1 | tee examples/cvpr17/log_exp.txt 

./build/tools/caffe train -solver=examples/cvpr17/exp_solver.prototxt -snapshot=models/cvpr17/ILT-exp_iter_101200.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log_exp2.txt 
## end up with iteration 186000
