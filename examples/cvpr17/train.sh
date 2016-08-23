#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=/home/lijun/Research/Code/FCT_scale_base/model/VGG_ILSVRC_16_layers.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log-smooth-7-1.txt 

./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=models/cvpr17-smooth/ILT-from-188000-5_iter_58000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log-smooth-7-1.txt 

# no major changes, continue training
#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17-smooth/ILT-from-188000-6_iter_1740.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log-smooth-6-2.txt 

# change conv6 and genmap learning rate to 100 and set max_value=1 for smooth pooling
#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17-smooth/ILT-from-188000-6_iter_16000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log-smooth-6-3.txt 

# no major changes. testing
#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17-smooth/ILT-from-188000-6_iter_18000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log-smooth-6-4.txt 

# change conv6 and genmap learning rate to 100000
#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -snapshot=models/cvpr17-smooth/ILT-from-188000-6_iter_18000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17/log-smooth-6-4.txt 
