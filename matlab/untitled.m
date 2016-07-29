caffe.set_mode_cpu();
net_model = ['../examples/smooth_pool/deploy.prototxt'];
net = caffe.Net(net_model, 'train');
input = 300*rand(27,27, 3, 10);
a = net.forward({single(input)});
a = a{1};
a = a(:);
b=max(max(input));
b = b(:);
caffe.reset_all
c = mean(mean(input));
c = c(:);
a (:,2) = b;
a(:, 3) = c;