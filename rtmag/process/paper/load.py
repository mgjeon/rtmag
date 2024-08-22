import torch
import numpy as np
import argparse
from pathlib import Path
from neuralop.models import UNO
from time import time
from rtmag.process.paper.diff import curl

def load_input_label(meta_path):
    meta_path = Path(meta_path)
    input_files = (meta_path / 'input').glob('*.npz')
    label_files = (meta_path / 'label').glob('*.npz')

    return sorted(list(input_files)), sorted(list(label_files))

class MyModel:
    def __init__(self, meta_path, epoch=None, device=None, clip=None, checkpo=None):
        self.meta_path = meta_path
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.epoch = epoch
        self.model, self.args, self.checkpoint = self.load_model(checkpo)
        self.clip = clip

    def load_model(self, checkpo=None):
        device = self.device
        meta_path = Path(self.meta_path)
        # if self.epoch is None:
        #     checkpoint = torch.load(meta_path / "best_model.pt", map_location=device)
        #     self.epoch = checkpoint['epoch']
        # else:
        #     try: 
        #         checkpoint = torch.load(meta_path / f"model_{self.epoch}.pt", map_location=device)
        #     except:
        #         checkpoint = torch.load(meta_path / "best_model.pt", map_location=device)
        #         self.epoch = checkpoint['epoch']
        checkpoint = torch.load(meta_path, map_location=device)
        self.epoch = checkpoint['epoch']

        args = argparse.Namespace()
        info = np.load(meta_path.parent / 'args.npy', allow_pickle=True).item()
        for key, value in info.items():
                args.__dict__[key] = value

        if args.model.get("factorization") is not None:
            model = UNO(
                    hidden_channels = args.model["hidden_channels"],
                    in_channels = args.model["in_channels"],
                    out_channels = args.model["out_channels"],
                    lifting_channels = args.model["lifting_channels"],
                    projection_channels = args.model["projection_channels"],
                    n_layers = args.model["n_layers"],

                    factorization = args.model["factorization"],
                    implementation = args.model["implementation"],
                    rank = args.model["rank"],

                    uno_n_modes = args.model["uno_n_modes"], 
                    uno_out_channels = args.model["uno_out_channels"],
                    uno_scalings = args.model["uno_scalings"],
                ).to(device)
        else:
            print("Model is not factorized")
            model = UNO(
                    hidden_channels = args.model["hidden_channels"],
                    in_channels = args.model["in_channels"],
                    out_channels = args.model["out_channels"],
                    lifting_channels = args.model["lifting_channels"],
                    projection_channels = args.model["projection_channels"],
                    n_layers = args.model["n_layers"],

                    uno_n_modes = args.model["uno_n_modes"], 
                    uno_out_channels = args.model["uno_out_channels"],
                    uno_scalings = args.model["uno_scalings"],
                ).to(device)

        if checkpo is None:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(checkpo, map_location=device))

        print(f"Model loaded from epoch {self.epoch}")

        return model, args, checkpoint


    def get_pred(self, input_path, b_norm=None):
        model = self.model
        args = self.args
        device = self.device
        model_input = np.load(input_path)['input'].astype(np.float32)
        model_input = torch.from_numpy(model_input)
        model_input = model_input[None, ...]
        # [batch_size, 3, 513, 257, 1]
        model_input = model_input[:, :, :-1, :-1, :]  # remove duplicated periodic boundary
        model_input = model_input.to(device)
        model_input = torch.permute(model_input, (0, 4, 3, 2, 1))
        
        clip = self.clip
        if clip is not None:
            model_input = torch.clip(model_input, -clip, clip)

        if b_norm is None:
            b_norm = args.data["b_norm"]
        elif b_norm == "max":
            b_norm = torch.max(torch.abs(model_input)).item()
        
        # [batch_size, 1, 256, 512, 3]
        model_input = model_input / b_norm
        self.model_input = model_input

        # [batch_size, 256, 256, 512, 3]
        model_output = model(model_input)
        self.outputs = model_output.detach().cpu()
        # [512, 256, 256, 3]
        b = model_output.detach().cpu().numpy().transpose(0, 3, 2, 1, 4)[0]
        divi = (b_norm / np.arange(1, b.shape[2] + 1)).reshape(1, 1, -1, 1)
        b = b * divi


        return b.astype(np.float32)
    
    def fine_tuning(self, dx=1, dy=1, dz=1, w_reg=0.01, w_bc=10, w_ff=1, w_div=1, lr=1e-5, n_epoch=100):
        from paper.diff_torch_batch import curl, divergence
        device = self.device

        model = self.model
        labels = self.outputs.to(device)
        model_input = self.model_input.to(device)
        B = torch.permute(model_input, (0, 3, 2, 1, 4))
        # [batch_size]
        dx = torch.from_numpy(np.array([dx]).astype(np.float32)).reshape(-1, 1).to(device)
        dy = torch.from_numpy(np.array([dy]).astype(np.float32)).reshape(-1, 1).to(device)
        dz = torch.from_numpy(np.array([dz]).astype(np.float32)).reshape(-1, 1).to(device)

        for i in range(n_epoch):
            model_output = model(model_input)

            # [batch_size, 256, 256, 512, 3]
            mse_fn = torch.nn.MSELoss()
            loss_mse = mse_fn(labels, model_output)

            # [b, z, y, x, 3] -> [b, x, y, z, 3]
            b = torch.permute(model_output, (0, 3, 2, 1, 4))
            
            # unnormalization
            divisor = (1 / np.arange(1, b.shape[-2] + 1)).reshape(1, 1, -1, 1).astype(np.float32)
            divisor = torch.from_numpy(divisor).to(device)
            b = b * divisor

            # Boundary Condition loss
            loss_bc = torch.mean(torch.square(b[:, :, :, 0, :] - B[:, :, :, 0, :]))

            # Force-freeness loss
            bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
            jx, jy, jz = curl(bx, by, bz, dx, dy, dz)
            b = torch.stack([bx, by, bz], -1)
            j = torch.stack([jx, jy, jz], -1)

            jxb = torch.cross(j, b, -1)
            loss_ff = (jxb**2).sum(-1) / ((b**2).sum(-1) + 1e-7)
            loss_ff = torch.mean(loss_ff)

            # Divergence-less loss
            div_b = divergence(bx, by, bz, dx, dy, dz)
            loss_div = torch.mean(torch.square(div_b))
                
            loss = w_reg*loss_mse + w_bc*loss_bc + w_ff*loss_ff + w_div*loss_div

            print(f"i = {i}, loss = {loss.item()}")

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_output = model(model_input)
        # [512, 256, 256, 3]
        b = model_output.detach().cpu().numpy().transpose(0, 3, 2, 1, 4)[0]
        b_norm = self.args.data["b_norm"]
        divi = (b_norm / np.arange(1, b.shape[2] + 1)).reshape(1, 1, -1, 1)
        b = b * divi

        return b.astype(np.float32)
    
    def get_pred_from_numpy(self, model_input, b_norm=None, ret=True):
        start_time = time()
        model = self.model
        args = self.args
        device = self.device
        if b_norm is None:
            b_norm = args.data["b_norm"]
        if ret:
            print(f"b_norm = {b_norm} G")
        # [bs, 1, 256, 512, 3]
        model_input = torch.from_numpy(model_input.astype(np.float32) / b_norm).to(device)
        model_output = model(model_input)
        self.model_input = model_input
        self.outputs = model_output.detach().cpu()

        # [512, 256, 256, 3]
        b = model_output.detach().cpu().numpy().transpose(0, 3, 2, 1, 4)[0]
        divi = (b_norm / np.arange(1, b.shape[-2] + 1)).reshape(1, 1, -1, 1)
        self.divi = divi
        b = b * divi

        if ret:
            print(f"{time() - start_time} seconds")
        return b.astype(np.float32)
    
    def get_pred_from_numpy_no_div(self, model_input, b_norm=None, ret=True):
        start_time = time()
        model = self.model
        args = self.args
        device = self.device
        if b_norm is None:
            b_norm = args.data["b_norm"]
        if ret:
            print(f"b_norm = {b_norm} G")
        # [bs, 1, 256, 512, 3]
        model_input = torch.from_numpy(model_input.astype(np.float32) / b_norm).to(device)
        model_output = model(model_input)
        self.model_input = model_input
        self.outputs = model_output.detach().cpu()

        # [512, 256, 256, 3]
        a = model_output.detach().cpu().numpy().transpose(0, 3, 2, 1, 4)[0]
        b = curl(a) * b_norm

        if ret:
            print(f"{time() - start_time} seconds")
        return b.astype(np.float32)
    

    def get_label(self, label_path):
        B = np.load(label_path)["label"][:, :-1, :-1, :-1].astype(np.float32)
        B = B.transpose(1, 2, 3, 0)
        clip = self.clip
        if clip is not None:
            B = np.clip(B, -clip, clip)

        return B.astype(np.float32)
    

    def get_pot(self, label_path):
        Bp = np.load(label_path)["pot"][:, :-1, :-1, :-1].astype(np.float32)
        Bp = Bp.transpose(1, 2, 3, 0)
        clip = self.clip
        if clip is not None:
            Bp = np.clip(Bp, -clip, clip)

        return Bp.astype(np.float32)
    
    def get_dV(self, input_path):
        inputs = np.load(input_path)
        dx, dy, dz = inputs['dx'], inputs['dy'], inputs['dz']  # Mm
        dx, dy, dz = dx * 1e8, dy * 1e8, dz * 1e8  # cm
        dV = dx * dy * dz # cm^3
        dV = dV.astype(np.float32)
        
        return dx, dy, dz, dV

    def get_coords(self, label_path):
        labels = np.load(label_path)
        x, y, z = labels['x'], labels['y'], labels['z']  # Mm
     
        return x, y, z
