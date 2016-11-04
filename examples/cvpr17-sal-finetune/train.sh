
./build/tools/caffe train -solver=examples/cvpr17-sal-finetune/solver.prototxt -weights=models/cvpr17-ILT-pretrain-fs/ip-1_iter_5000.caffemodel -gpu=0,1 2>&1  | tee examples/cvpr17-sal-finetune/log-2-0.txt
#resotre from iter 94000
#./build/tools/caffe train -solver=examples/cvpr17-sal-finetune/solver.prototxt -weights=models/cvpr17-sal-finetune/sal-finetune-1-0_iter_42500.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17-sal-finetune/log-1-1.txt 

#resotre from iter 102500
#./build/tools/caffe train -solver=examples/cvpr17-ILT-pretrain-exp2/solver.prototxt -snapshot=models/cvpr17-ILT-pretrain-exp2/ip-1_iter_102500.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17-ILT-pretrain-exp2/log-ip-1-3.txt 

#restore from iter 80000, decrease lr by 0.1
#./build/tools/caffe train -solver=examples/cvpr17-ILT-pretrain-exp2/solver.prototxt -snapshot=models/cvpr17-ILT-pretrain-exp2/ip-1_iter_14000.solverstate -gpu=0,1 2>&1 | tee examples/cvpr17-ILT-pretrain-exp2/log-ip-2.txt 


#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=models/cvpr17-ILT/ILT-_iter_44000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log-ILT-3.txt 

#./build/tools/caffe train -solver=examples/cvpr17/solver.prototxt -weights=models/cvpr17-ILT/ILT-3_iter_74000.caffemodel -gpu=0,1 2>&1 | tee examples/cvpr17/log-ILT-4.txt 
