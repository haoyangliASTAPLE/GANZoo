# GANZoo
 
### Environment Setup
Install Python 3.X and Pytorch according to your CUDA version
Pip install `ema-pytorch~=0.2.3` and `tqdm`

### Train a GAN
For now, this code only implements training Wasserstein GAN with Gradient Penalty using MNIST, run this experiment by running the `wgan.py` file. You can call `python wgan.py --params1 k1 --params2 k2` in terminal to customize your own parameters, where `params1` or `params2` are the parameter name and `k1` or `k2` are the parameter values.