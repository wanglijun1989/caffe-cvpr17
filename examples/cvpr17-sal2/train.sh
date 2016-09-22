
#./build/tools/caffe train -solver=examples/cvpr17-sal2/solver.prototxt -weights=models/cvpr17-ILT/ILT-3_iter_6000.caffemodel   2>&1 | tee examples/cvpr17-sal2/log-sal2-1.txt 
#resotre from iter  3100, lr = 0.0001
./build/tools/caffe train -solver=examples/cvpr17-sal2/solver.prototxt -snapshot=models/cvpr17-sal2/sal-1_iter_3100.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17-sal2/log-sal2-2.txt 
# restore from iter 57000
#./build/tools/caffe train -solver=examples/cvpr17-sal/solver.prototxt -snapshot=models/cvpr17-sal/sal-1_iter_57000.solverstate -gpu=0 2>&1 | tee examples/cvpr17-sal/log-sal-3.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=models/cvpr17-ILT/ILT-_iter_44000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log-ILT-3.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=models/cvpr17-ILT/ILT-3_iter_74000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log-ILT-4.txt 
