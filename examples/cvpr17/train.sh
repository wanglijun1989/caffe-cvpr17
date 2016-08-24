
#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=models/cvpr17-smooth/ILT-from-188000-5_iter_58000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log-smooth-7-1.txt 

#change max_value in smooth_pooling from 32 to 64
./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17-smooth/ILT-from-188000-7_iter_6000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log-smooth-7-2.txt 

