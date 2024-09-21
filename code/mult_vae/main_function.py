import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from scipy import sparse

from . import vae_models


def mult_vae_inference(bi_graph: sparse._csr.csr_matrix) -> torch.Tensor:
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='C:/codes/buffer/vae_model.pt',
                        help='path to save the final model')
    args = parser.parse_args()

    # Set the random seed manually for reproductibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")


    # Load data
    train_data = bi_graph
    user_array = torch.Tensor(bi_graph.toarray()).to(device)
    n_users = train_data.shape[0]
    n_items = train_data.shape[1]
    idxlist = list(range(n_users))

    # Build the model
    p_dims = [200, 600, n_items]
    model = vae_models.MultiVAE(p_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
    criterion = vae_models.loss_function

    # Training code
    # TensorboardX Writer
    writer = SummaryWriter()

    best_n100 = -np.inf
    global update_count
    update_count = 0

    def naive_sparse2tensor(data):
        return torch.FloatTensor(data.toarray())

    def train():
        # Turn on training mode
        model.train()
        train_loss = 0.0
        start_time = time.time()
        global update_count

        np.random.shuffle(idxlist)
        
        for batch_idx, start_idx in enumerate(range(0, n_users, args.batch_size)):
            end_idx = min(start_idx + args.batch_size, n_users)
            data = train_data[idxlist[start_idx:end_idx]]
            data = naive_sparse2tensor(data).to(device)

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 
                                1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            loss = criterion(recon_batch, data, mu, logvar, anneal)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            update_count += 1

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                        'loss {:4.2f}'.format(
                            epoch, batch_idx, len(range(0, n_users, args.batch_size)),
                            elapsed * 1000 / args.log_interval,
                            train_loss / args.log_interval))
                
                # Log loss to tensorboard
                n_iter = (epoch - 1) * len(range(0, n_users, args.batch_size)) + batch_idx
                writer.add_scalars('data/loss', {'train': train_loss / args.log_interval}, n_iter)

                start_time = time.time()
                train_loss = 0.0

    def get_user_emb():
        # user_array = torch.Tensor(bi_graph.toarray()).to(device)
        user_emb_mu = torch.zeros([n_users, 200], dtype=torch.float32, device=device)
        user_emb_std = torch.zeros([n_users, 200], dtype=torch.float32, device=device)
        for i in range(user_array.shape[0]):
            user_emb_mu[i:i+1, :], user_emb_std[i:i+1, :] = model.encode_only(user_array[i:i+1, :])
        return user_emb_mu, user_emb_std

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with open(args.save, 'rb') as f:
            model = torch.load(f)
    except:
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            train()
        with open(args.save, 'wb') as f:
            torch.save(model, f)

    user_emb_mu, user_emb_std = get_user_emb()

    return user_emb_mu, user_emb_std
