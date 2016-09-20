
./build/tools/caffe train -solver=examples/cvpr17-sal-exp/solver.prototxt -weights=models/cvpr17-ILT/ILT-3_iter_6000.caffemodel -gpu=1  2>&1 | tee examples/cvpr17-sal-exp/log-sal-1.txt 
#resotre from iter , snapshot interval change to 1000
#./build/tools/caffe train -solver=examples/cvpr17-sal/solver.prototxt -snapshot=models/cvpr17-sal/sal-1_iter_4000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17-sal/log-sal-2.txt 
# restore from iter 57000
#./build/tools/caffe train -solver=examples/cvpr17-sal/solver.prototxt -snapshot=models/cvpr17-sal/sal-1_iter_57000.solverstate -gpu=0 2>&1 | tee examples/cvpr17-sal/log-sal-3.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=models/cvpr17-ILT/ILT-_iter_44000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log-ILT-3.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=models/cvpr17-ILT/ILT-3_iter_74000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log-ILT-4.txt 
