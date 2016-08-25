
#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=/home/lijun/Research/Code/CVPR17/model/vgg16CAM_train_iter_90000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log-ILT-2.txt 
#resotre from iter 115000
./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17-ILT/ILT_iter_115000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log-ILT-2.txt 

