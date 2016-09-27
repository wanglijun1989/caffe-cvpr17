
./build/tools/caffe train -solver=examples/cvpr17-sal4/solver.prototxt -weights=models/cvpr17-ILT/ILT-3_iter_6000.caffemodel -gpu=1  2>&1 | tee examples/cvpr17-sal4/log-sal4-1.txt 
#resotre from iter  9000, lr = 0.0001
#./build/tools/caffe train -solver=examples/cvpr17-sal3/solver.prototxt -snapshot=models/cvpr17-sal3/sal-1_iter_9000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17-sal3/log-sal3-2.txt 

#resotre from iter  25000, lr = 0.0001
#./build/tools/caffe train -solver=examples/cvpr17-sal3/solver.prototxt -snapshot=models/cvpr17-sal3/sal-1_iter_25000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17-sal3/log-sal3-2-2.txt 
