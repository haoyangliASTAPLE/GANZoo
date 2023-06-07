import torch
from torch import nn

# ---------------Weight Initialization---------------
def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0, 0.01)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


# ---------------Deep Convolutional GAN Generator---------------
class Generator(nn.Module):
    def __init__(self, in_dim=100, out_channel=3, dim=64):
        super(Generator, self).__init__()
        # This create one layer
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())

        # We use a generator with 4 deconv layer. The last layer doesn't have 
        # BatchNorm and ReLU
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, out_channel, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())

        # Initialize the weight
        self.apply(init_weights)

    def forward(self, x):
        # When we use G = Generator() and G(z), this function will be called
        # and will return the output
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


# ---------------Deep Convolutional GAN Discriminator---------------
class Discriminator(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(Discriminator, self).__init__()

        # This create one layer
        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        # We use a discriminator with 4 conv layer
        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim*2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim*2, dim*4, 5, 2, 2)
        self.layer4 = conv_ln_lrelu(dim*4, dim*4, 3, 2, 1)

        self.fc_layer = nn.Linear(dim*4*4*4+1, 1)

        self.apply(init_weights)

    def forward(self, x):
        # When we use D = Discriminator() and G(x), this function will 
        # be called and will return the output
        bs = x.shape[0]
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat4 = feat4.view(bs, -1)

        # Here we use Minibatch Standard Deviation. 
        # It alleviates mode collapse to some extend.
        batch_std = torch.std(feat4, dim=0)
        mb_std = torch.mean(batch_std).repeat(bs, 1)
        feat4 = torch.cat([feat4, mb_std], dim=1)

        y = self.fc_layer(feat4)

        return y
