import argparse
import os
from ema_pytorch import EMA
from tqdm import tqdm
import torch.cuda
from torch import nn, optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from models import Generator, Discriminator

# freeze the parameter
def freeze(net: nn.Module):
    for p in net.parameters():
        p.requires_grad_(False)

# unfreeze the parameter
def unfreeze(net: nn.Module):
    for p in net.parameters():
        p.requires_grad_(True)

def gradient_penalty(x, y, discriminator):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = discriminator(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create save directory
    save_dir_img = './.saved_figs/' + args.dataset
    save_dir_model = './.saved_models/' + args.dataset
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)
    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)

    if args.dataset == 'mnist':
        # normalize function
        mean = (0.1307,)
        std = (0.3081,)
        normalize = transforms.Normalize(mean, std)

        # load MNIST dataset, we use training set as the auxiliary dataset
        train = datasets.MNIST(train=True, transform=transforms.Compose([transforms.ToTensor, normalize]))
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

        # dataset characteristics
        channel = 1
        size = 28

    if args.arch == 'dcgan':
        # create model
        generator = Generator(out_channel=channel, dim=size).to(device)
        discriminator = Discriminator(in_dim=channel, dim=size).to(device)

    # exponential moving average
    ema = None
    if args.ema:
        ema = EMA(generator, beta=0.999).to(device)

    # optimizer
    if args.optim == 'Adam':
        optimizer_g = optim.Adam(generator.parameters(), 
                               lr=args.lr, 
                               betas=args.betas)
        optimizer_d = optim.Adam(discriminator.parameters(), 
                               lr=args.lr, 
                               betas=args.betas)

    # training
    print('start training')
    step = 0
    for epoch in range(1, args.epochs + 1):
        for real, _ in tqdm(loader):
            step += 1
            real = real.to(device)
            bs = real.size(0)

            # train the discriminator
            freeze(generator)
            unfreeze(discriminator)

            # clear any previously stored gradients
            optimizer_d.zero_grad()

            # sample some latent codes from Gaussian and generate fake images
            z = torch.randn(bs, args.z_dim).to(device)
            fake = generator(z)
            fake = normalize(fake)

            # output
            logits_real = discriminator(real)
            logits_fake = discriminator(fake)

            # Wasserstein-1 Distance
            wd = logits_real.mean() - logits_fake.mean()
            # gradient penalty
            gp = gradient_penalty(real.data, fake.data, discriminator)

            # loss function
            loss_d = - wd + gp * 10.0

            # calculate gradient
            loss_d.backward()
            # clip the gradient if grad_clip is set
            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=args.grad_clip)
            # update parameter
            optimizer_d.step()

            if step % args.n_critic == 0:
                # train the generator
                freeze(discriminator)
                unfreeze(generator)

                # loss function for generator
                loss_g = -logits_fake.mean()

                loss_g.backward()
                if args.grad_clip is not None:
                    nn.utils.clip_grad_norm_(generator.parameters(), max_norm=args.grad_clip)
                optimizer_g.step()

                # update ema if ema is set
                if ema is not None:
                    ema.update()

        print(f'epoch {epoch}, G loss {loss_g}, D loss {loss_d}')

        # save fake images
        if epoch % args.eval_steps == 0:
            z = torch.randn(32, args.z_dim).to(device)
            with torch.no_grad():
                fake = ema(z) if ema is not None else generator(z)
            save_image(fake.detach(), f"{save_dir_img}/image_epoch_{epoch}.jpg", 
                       normalize=False, nrow=8, padding=0)

        # save models
        if epoch % args.save_steps == 0:
            save_path = f'{save_dir_model}/{args.arch}_generator.pth'
            torch.save(ema.ema_model.state_dict() if ema is not None \
                       else generator.state_dict(), save_path)
            save_path = f'{save_dir_model}/{args.arch}_discriminator.pth'
            torch.save(discriminator.state_dict(), save_path)

if __name__ == '__main__':
    # We use argument parser to get the input parameter from the user. 
    # Change a different parameter by calling python wgan.py --param k,
    # where 'param' is the name of the parameter and k is the value.
    parser = argparse.ArgumentParser()

    # model and dataset setting
    parser.add_argument('--arch', type=str, default='dcgan', help='architecture of GAN')
    parser.add_argument('--dataset', type=str, default='mnist')

    # optimization setting
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='factor to contrl l2-norm regularization')
    parser.add_argument('--betas', type=tuple, default=(0.5, 0.999), help='args for Adam optimizer')
        # exponential moving avg
    parser.add_argument('--ema', type=int, default=1, help='exponential moving avg')
        # magnitude of gradient clipping (None for no clipping)
    parser.add_argument('--grad_clip', type=float, default=None, help='magnitude of gradient clipping')
    parser.add_argument('--n_critic', type=int, default=5, help='how often to update G')
    parser.add_argument('--z_dim', type=int, default=100, help='dimension of latent code')

    # training setting
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_steps', type=int, default=10, help='how often to evaluate fake imgs')
    parser.add_argument('--save_steps', type=int, default=10, help='how often to save model')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers for data loaders')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin the memory for train loader')

    args = parser.parse_args()
    main(args)