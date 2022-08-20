import torch as ch

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, act_fn=ch.nn.ReLU):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            act_fn(inplace=True)
    )

def BasicConvModel(n_channels, num_classes, act_fn=ch.nn.ReLU):
    model_name = f"BasicConvModel-{act_fn.__name__}"
    model = ch.nn.Sequential(
        conv_bn(n_channels, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        conv_bn(256, 256),
        conv_bn(256, 256),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_classes, bias=False),
    )
    model = model.to(memory_format=ch.channels_last).cuda()
    return model, model_name