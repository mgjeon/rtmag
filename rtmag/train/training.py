import os
import gc

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
from torchmetrics.regression import ConcordanceCorrCoef, MeanSquaredError

from rtmag.dataset.dataset_hnorm_unit_aug import ISEEDataset_Multiple_Hnorm_Unit_Aug, ISEEDataset_Hnorm_Unit_Aug
from rtmag.dataset.dataset_hnorm_unit_lowlou import LowLouDataset_Multiple_Hnorm_Unit

from rtmag.utils.diff_torch_batch import curl, divergence
from rtmag.utils.eval_plot import plot_sample
from rtmag.utils import eval

if torch.cuda.is_available():
    device = torch.device("cuda")


#---------------------------------------------------------------------------------------
def get_dataloaders(args):
    if args.data["dataset_name"] == "Hnorm_Unit_Aug":
        train_dataset = ISEEDataset_Multiple_Hnorm_Unit_Aug(args.data['dataset_path'], args.data["b_norm"], exc_noaa=args.data['exc_noaa'])
        val_dataset = ISEEDataset_Hnorm_Unit_Aug(args.data['val_path'], args.data["b_norm"])
    elif args.data["dataset_name"] == "LowLou_Hnorm_Unit":
        train_dataset = LowLouDataset_Multiple_Hnorm_Unit(args.data['train_path'], args.data["b_norm"])
        val_dataset = LowLouDataset_Multiple_Hnorm_Unit(args.data['val_path'], args.data["b_norm"])
    else:
        raise NotImplementedError
    
    train_dataloder = DataLoader(train_dataset, batch_size=args.data['batch_size'], shuffle=True, 
                                 num_workers=args.data["num_workers"], pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                 num_workers=args.data["num_workers"], pin_memory=True)

    return train_dataloder, val_dataloader


#---------------------------------------------------------------------------------------
def shared_step(model, sample_batched, args):
    # input --------------------------------------------------------------
    # [b, 3, x, y, 1]
    inputs = sample_batched['input'].to(device)
    # [b, 3, x, y, 1] -> [b, 1, y, x, 3]
    inputs = torch.permute(inputs, (0, 4, 3, 2, 1))

    # predict ------------------------------------------------------------
    # [b, 1, y, x, 3] -> [b, 256, y, x, 3]
    outputs = model(inputs).to(device)

    # label --------------------------------------------------------------
    # [b, 3, x, y, z]
    labels = sample_batched['label'].to(device)

    # [b, 3, x, y, z] -> [b, z, y, x, 3]
    labels = torch.permute(labels, (0, 4, 3, 2, 1))

    # [b]
    dx = sample_batched['dx'].flatten().to(device)
    dy = sample_batched['dy'].flatten().to(device)
    dz = sample_batched['dz'].flatten().to(device)

    loss = criterion(outputs, labels, dx, dy, dz, args)

    return loss


#---------------------------------------------------------------------------------------
def criterion(outputs, labels, dx, dy, dz, args):

    loss = {}
    
    # [b, z, y, x, 3] -> [b, z, ...]
    opts = torch.flatten(outputs, start_dim=2)
    lbls = torch.flatten(labels, start_dim=2)

    # [b, z, ...] -> [b, ..., z]
    opts = torch.permute(opts, (0, 2, 1))
    lbls = torch.permute(lbls, (0, 2, 1))

    # mse loss
    mse = MeanSquaredError(num_outputs=opts.shape[-1]).to(device)
    loss_mse = 0.0
    for i in range(opts.shape[0]):
        loss_mse += torch.mean(mse(opts[i], lbls[i]))
    loss_mse /= opts.shape[0]
    loss['mse'] = loss_mse

    # ccc loss
    ccc = ConcordanceCorrCoef(num_outputs=opts.shape[-1]).to(device)
    loss_ccc = 0.0
    for i in range(opts.shape[0]):
        loss_ccc += torch.mean(torch.square(1.0 - ccc(opts[i], lbls[i])))
    loss_ccc /= opts.shape[0]
    loss['ccc'] = loss_ccc

    # [b, z, y, x, 3] -> [b, x, y, z, 3]
    b = torch.permute(outputs, (0, 3, 2, 1, 4))
    B = torch.permute(labels, (0, 3, 2, 1, 4))

    # denormalization
    divisor = (1 / np.arange(1, b.shape[-2] + 1)).reshape(1, 1, -1, 1).astype(np.float32)
    divisor = torch.from_numpy(divisor).to(device)
    b = b * divisor
    B = B * divisor

    # bc loss
    # bottom (z=0)
    b_bottom = b[:, :, :, 0, :].flatten()
    B_bottom = B[:, :, :, 0, :].flatten()

    mse = MeanSquaredError().to(device)
    loss['bc_bottom'] = mse(b_bottom, B_bottom)

    # PDE loss
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    jx, jy, jz = curl(bx, by, bz, dx, dy, dz)
    b = torch.stack([bx, by, bz], -1)
    j = torch.stack([jx, jy, jz], -1)

    # force-free loss
    jxb = torch.cross(j, b, -1)
    loss_ff = (jxb**2).sum(-1) / ((b**2).sum(-1) + 1e-7)
    loss_ff = torch.mean(loss_ff)
    loss['ff'] = loss_ff

    # divergence-free loss
    div_b = divergence(bx, by, bz, dx, dy, dz)
    loss_div = torch.mean(torch.square(div_b))
    loss['div'] = loss_div

    return loss


#---------------------------------------------------------------------------------------
def eval_plots(b_pred, b_true, b_pot, func, name):
    heights = np.arange(b_pred.shape[-2])

    plots_b = []
    for i in range(b_pred.shape[-2]):
        plots_b.append(func(b_pred[:, :, i, :], b_true[:, :, i, :]))

    if b_pot is not None:
        plots_B_pot = []
        for i in range(b_pot.shape[-2]):
            plots_B_pot.append(func(b_pot[:, :, i, :], b_true[:, :, i, :]))

    fig = plt.figure(figsize=(6, 8))
    plt.plot(plots_b, heights, color='red', label='PINO')
    if b_pot is not None:
        plt.plot(plots_B_pot, heights, color='black', label='Potential')
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('height [pixel]')
    plt.xscale('log')
    plt.yscale('linear')
    plt.grid()
    plt.tight_layout()
    return fig

#---------------------------------------------------------------------------------------
def val_plot(model, val_dataloader, epoch, args, writer):
    with torch.no_grad():
        batch = next(iter(val_dataloader))

        b_norm = args.data["b_norm"]

        # [b, 3, x, y, 1]
        inputs = batch['input'].to(device)
        # [b, 3, x, y, 1] -> [b, 1, y, x, 3]
        inputs = torch.permute(inputs, (0, 4, 3, 2, 1))
        # [b, 1, y, x, 3] -> [b, z, y, x, 3]
        outputs = model(inputs)

        # [b, z, y, x, 3] -> [x, y, z, 3]
        b_pred = outputs.detach().cpu().numpy()
        b_pred = b_pred[0, ...].transpose(2, 1, 0, 3)

        # [b, 3, x, y, z] -> [x, y, z, 3]
        b_true = batch['label'].detach().cpu().numpy()
        b_true = b_true[0, ...].transpose(1, 2, 3, 0)

        # denormalization
        divi = (b_norm / np.arange(1, b_pred.shape[-2] + 1)).reshape(1, 1, -1, 1).astype(np.float32)
        b_pred = b_pred * divi
        b_true = b_true * divi

        fig1, fig2 = plot_sample(b_pred, b_true, ret=True, v_mm=b_norm/2)
        writer.add_figure('plot/pred', fig1, epoch)
        writer.add_figure('plot/true', fig2, epoch)
        plt.close()
        
        #-----------------------------------------------------------
        if 'pot' in batch.keys():
            b_pot = batch['pot'].detach().cpu().numpy()
            b_pot = b_pot[0, ...].transpose(1, 2, 3, 0)
            b_pot = b_pot * divi
        else:
            b_pot = None
    
        fig = eval_plots(b_pred, b_true, b_pot, eval.l2_error, 'rel_l2_err')
        writer.add_figure(f'plot/rel_l2_err', fig, epoch)
        plt.close()

        fig = eval_plots(b_pred, b_true, b_pot, eval.eps, 'eps')
        writer.add_figure(f'plot/eps', fig, epoch)
        plt.close()

        gc.collect()
        torch.cuda.empty_cache()


#---------------------------------------------------------------------------------------
def val(model, val_dataloader, epoch, args, writer):
    with torch.no_grad():

        total_val_loss = 0.0
        total_val_loss_mse = 0.0
        total_val_loss_ccc = 0.0
        total_val_loss_bc_bottom = 0.0
        total_val_loss_ff = 0.0
        total_val_loss_div = 0.0

        for i_batch, sample_batched in enumerate(tqdm(val_dataloader, position=1, desc='Validation', leave=False, ncols=70)):
            gc.collect()
            torch.cuda.empty_cache()
            
            val_loss_dict = shared_step(model, sample_batched, args)

            val_loss = args.training['w_mse']*val_loss_dict['mse'] \
                     + args.training['w_ccc']*val_loss_dict['ccc'] \
                     + args.training['w_bc_bottom']*val_loss_dict['bc_bottom'] \
                     + args.training['w_ff']*val_loss_dict['ff'] \
                     + args.training['w_div']*val_loss_dict['div']

            total_val_loss += val_loss.item()
            total_val_loss_mse += val_loss_dict['mse'].item()
            total_val_loss_ccc += val_loss_dict['ccc'].item()
            total_val_loss_bc_bottom += val_loss_dict['bc_bottom'].item()
            total_val_loss_ff += val_loss_dict['ff'].item()
            total_val_loss_div += val_loss_dict['div'].item()
        
        total_val_loss /= len(val_dataloader)
        total_val_loss_mse /= len(val_dataloader)
        total_val_loss_ccc /= len(val_dataloader)
        total_val_loss_bc_bottom /= len(val_dataloader)
        total_val_loss_ff /= len(val_dataloader)
        total_val_loss_div /= len(val_dataloader)

        writer.add_scalar('val/loss', total_val_loss, epoch)
        writer.add_scalar('val/loss_mse', total_val_loss_mse, epoch)
        writer.add_scalar('val/loss_ccc', total_val_loss_ccc, epoch)
        writer.add_scalar('val/loss_bc_bottom', total_val_loss_bc_bottom, epoch)
        writer.add_scalar('val/loss_ff', total_val_loss_ff, epoch)
        writer.add_scalar('val/loss_div', total_val_loss_div, epoch)

        return total_val_loss


#---------------------------------------------------------------------------------------
def train(model, optimizer, train_dataloader, val_dataloader, ck_epoch, CHECKPOINT_PATH, args, writer, scheduler=None):
    model = model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    base_path = args.base_path
    n_epochs = args.training['n_epochs']
    
    global_step_tmp = len(train_dataloader) * ck_epoch
    validation_loss = np.inf
    for epoch in range(ck_epoch, n_epochs+1):

        if epoch == 0:
            with torch.no_grad():
                model = model.eval()
                val_plot(model, val_dataloader, -1, args, writer)

        # Training
        model = model.train()
        total_train_loss = 0.0
        total_train_loss_mse = 0.0
        total_train_loss_ccc = 0.0
        total_train_loss_bc_bottom = 0.0
        total_train_loss_ff = 0.0
        total_train_loss_div = 0.0

        with tqdm(train_dataloader, desc='Training', ncols=140) as tqdm_loader_train:
            for i_batch, sample_batched in enumerate(tqdm_loader_train):
                gc.collect()
                torch.cuda.empty_cache()

                tqdm_loader_train.set_description(f"epoch {epoch}")
                
                loss_dict = shared_step(model, sample_batched, args)

                loss = args.training['w_mse']*loss_dict['mse'] \
                     + args.training['w_ccc']*loss_dict['ccc'] \
                     + args.training['w_bc_bottom']*loss_dict['bc_bottom'] \
                     + args.training['w_ff']*loss_dict['ff'] \
                     + args.training['w_div']*loss_dict['div']
        
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                global_step = global_step_tmp + epoch + i_batch
                tqdm_loader_train.set_postfix(step=global_step)

                total_train_loss += loss.item()
                total_train_loss_mse += loss_dict['mse'].item()
                total_train_loss_ccc += loss_dict['ccc'].item()
                total_train_loss_bc_bottom += loss_dict['bc_bottom'].item()
                total_train_loss_ff += loss_dict['ff'].item()
                total_train_loss_div += loss_dict['div'].item()

                writer.add_scalar('step_train/loss', loss.item(), global_step)
                writer.add_scalar('step_train/loss_mse', loss_dict['mse'].item(), global_step)
                writer.add_scalar('step_train/loss_ccc', loss_dict['ccc'].item(), global_step)
                writer.add_scalar('step_train/loss_bc_bottom', loss_dict['bc_bottom'].item(), global_step)
                writer.add_scalar('step_train/loss_ff', loss_dict['ff'].item(), global_step)
                writer.add_scalar('step_train/loss_div', loss_dict['div'].item(), global_step)

                writer.add_scalar('epoch', epoch, global_step)
                
                torch.save({'epoch': epoch, 'global_step': global_step, 
                            'model_state_dict': model.state_dict()}, 
                            os.path.join(args.base_path, "last_model.pt"))

        total_train_loss /= len(train_dataloader)
        total_train_loss_mse /= len(train_dataloader)
        total_train_loss_ccc /= len(train_dataloader)
        total_train_loss_bc_bottom /= len(train_dataloader)
        total_train_loss_ff /= len(train_dataloader)
        total_train_loss_div /= len(train_dataloader)

        writer.add_scalar('train/loss', total_train_loss, epoch)
        writer.add_scalar('train/loss_mse', total_train_loss_mse, epoch)
        writer.add_scalar('train/loss_ccc', total_train_loss_ccc, epoch)
        writer.add_scalar('train/loss_bc_bottom', total_train_loss_bc_bottom, epoch)
        writer.add_scalar('train/loss_ff', total_train_loss_ff, epoch)
        writer.add_scalar('train/loss_div', total_train_loss_div, epoch)
        
        global_step_tmp = global_step

        torch.save({'epoch': epoch, 'global_step': global_step, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    CHECKPOINT_PATH)
        
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.training['save_epoch_every'] == 0:
            torch.save({'epoch': epoch, 'global_step': global_step,
                        'model_state_dict': model.state_dict()}, os.path.join(base_path, f"model_{epoch}.pt"))

        # Validation
        with torch.no_grad():
            model = model.eval()
            val_plot(model, val_dataloader, epoch, args, writer)
            total_val_loss = val(model, val_dataloader, epoch, args, writer)

        if os.path.exists(os.path.join(base_path, "best_model.pt")):
            checkpoint = torch.load(os.path.join(base_path, "best_model.pt"))
            validation_loss = checkpoint['validation_loss']

        if total_val_loss < validation_loss:
            validation_loss = total_val_loss
            torch.save({'epoch': epoch, 'global_step': global_step, 'validation_loss': validation_loss,
                        'model_state_dict': model.state_dict()}, os.path.join(base_path, "best_model.pt"))