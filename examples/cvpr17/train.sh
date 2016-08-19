#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=/home/lijun/Research/Code/CVPR17/model/vgg16CAM_train_iter_90000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log1.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -gpu=0,1 2>&1 | tee examples/cvpr17/log1.txt 


#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_128000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log2.txt 
## end up with iteration 100000

## negtive scale = 0.05,
#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_100000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log3.txt 
## Add overlap_accuracy layer
##./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_126000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log4.txt 
## Resume from iteration: 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_284000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log5.txt 

# finetuning vgg16
#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=/home/lijun/Research/Code/CVPR17/model/vgg16CAM_train_iter_90000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log2-1.txt 

# decrease learning rate at iteration 188000
#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17-finetune/ILT_iter_188000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log2-2.txt 

# add generic map layer
./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=models/cvpr17-finetune/ILT_iter_188000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log3-1.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_2000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log2.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_76000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log3.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17/ILT_iter_78000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log4.txt 
