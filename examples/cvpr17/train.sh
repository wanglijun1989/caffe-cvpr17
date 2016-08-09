#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=/home/lijun/Research/Code/CVPR17/model/vgg16CAM_train_iter_90000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log1.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -gpu=0,1 2>&1 | tee examples/cvpr17/log1.txt 


./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_128000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log2.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_2000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log2.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_76000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log3.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_78000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log4.txt 
