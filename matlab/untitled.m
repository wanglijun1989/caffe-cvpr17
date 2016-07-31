caffe.set_mode_cpu();
net_model = ['../examples/smooth_pool/deploy.prototxt'];
net = caffe.Net(net_model, 'train');
input = 300*rand(10, 10, 10, 10);
out_diff = rand(1, 1, 10, 10);
ave_pool = mean(mean(input));
ave_pool = ave_pool(:);
max_pool=max(max(input));
max_pool = max_pool(:);
tic;
out = net.forward({single(input)});
net.backward({single(out_diff)});
toc
out = out{1};
out = out(:);
% caffe.reset_all
res = out;
% conv_diff1 = net.blobs('conv1').get_diff();
data_diff1 = net.blobs('data').get_diff();
caffe.set_mode_gpu();
% net_model = ['../examples/smooth_pool/deploy.prototxt'];
% net = caffe.Net(net_model, 'train');
tic;
out = net.forward({single(input)});
net.backward({single(out_diff)});
toc
% conv_diff2 = net.blobs('conv1').get_diff();
data_diff2 = net.blobs('data').get_diff();
out = out{1};
out = out(:);
res(:,2) = out;
res(:,3) = ave_pool;
res(:,4) = max_pool;
% res_diff(:, 1) = conv_diff1(:);
% res_diff(:, 2) = conv_diff2(:);
res2_diff(:, 1) = data_diff1(:);
res2_diff(:, 2) = data_diff2(:);
caffe.reset_all