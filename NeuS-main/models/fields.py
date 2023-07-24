import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
import math


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

        self.attention = SelfAttention(dims[0])

    def forward(self, inputs):

        inputs = inputs.unsqueeze(1)
        inputs = self.attention(inputs)
        inputs = inputs.squeeze(1)

        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        # print(f'inputs.shape: {inputs.shape}')
        # print(f'x.shape: {x.shape}')
        # reshape
        # x = x.unsqueeze(1)
        # print(f'x.shape unsq: {x.shape}')
        # x = self.attention(x)
        # x = x.squeeze(1)

        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

# class SDFNetwork(nn.Module):
#     def __init__(self, d_in, d_out, d_hidden, n_layers, skip_in=(4,), multires=0, bias=0.5, scale=1,
#                  geometric_init=True, weight_norm=True, inside_outside=False):
#         super(SDFNetwork, self).__init__()
#
#         self.num_layers = n_layers
#         self.skip_in = skip_in
#         self.scale = scale
#         self.d_k = d_hidden
#         self.latent_dim = 512
#
#         # Encoder layers
#         self.encoder = nn.ModuleList()
#         for i in range(n_layers):
#             if i == 0:
#                 self.encoder.append(nn.Linear(d_in, d_hidden))
#             else:
#                 self.encoder.append(nn.Linear(d_hidden, d_hidden))
#
#         self.encoder.append(nn.Linear(d_hidden, self.latent_dim))
#
#         # Decoder layers
#         self.decoder = nn.ModuleList()
#
#         self.decoder.append(nn.Linear(self.latent_dim, d_hidden))
#
#         for i in range(n_layers):
#             if i == n_layers - 1:
#                 self.decoder.append(nn.Linear(d_hidden, d_out))
#             else:
#                 self.decoder.append(nn.Linear(d_hidden, d_hidden))
#
#         # Attention mechanism
#         self.attention = nn.Linear(self.latent_dim, self.latent_dim)
#
#         self.activation = nn.Softplus(beta=100)
#
#         self.attention = SelfAttention(self.latent_dim)
#
#     def forward(self, inputs):
#         # Encoding
#         x = inputs
#         for layer in self.encoder:
#             x = self.activation(layer(x))
#
#         # Reshape x for attention
#         # x = x.view(x.size(0), -1, self.latent_dim)
#         #
#         # x = self.attention(x)
#         # x = x.view(x.size(0), -1)
#
#         # Decoding
#         for layer in self.decoder:
#             x = self.activation(layer(x))
#
#         return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    # This implementation is borrowed from IDR: https://github.com/lioryariv/idr
# class SDFNetwork(nn.Module):
#     def __init__(self,
#                  d_in,
#                  d_out,
#                  d_hidden,
#                  n_layers,
#                  skip_in=(4,),
#                  multires=0,
#                  bias=0.5,
#                  scale=1,
#                  geometric_init=True,
#                  weight_norm=True,
#                  inside_outside=False):
#         super(SDFNetwork, self).__init__()
#
#         dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
#
#         self.embed_fn_fine = None
#
#         if multires > 0:
#             embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
#             self.embed_fn_fine = embed_fn
#             dims[0] = input_ch
#
#         self.num_layers = len(dims)
#         self.skip_in = skip_in
#         self.scale = scale
#
#         for l in range(0, self.num_layers - 1):
#             if l + 1 in self.skip_in:
#                 out_dim = dims[l + 1] - dims[0]
#             else:
#                 out_dim = dims[l + 1]
#
#             lin = nn.Linear(dims[l], out_dim)
#
#             if geometric_init:
#                 if l == self.num_layers - 2:
#                     if not inside_outside:
#                         torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
#                         torch.nn.init.constant_(lin.bias, -bias)
#                     else:
#                         torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
#                         torch.nn.init.constant_(lin.bias, bias)
#                 elif multires > 0 and l == 0:
#                     torch.nn.init.constant_(lin.bias, 0.0)
#                     torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
#                     torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
#                 elif multires > 0 and l in self.skip_in:
#                     torch.nn.init.constant_(lin.bias, 0.0)
#                     torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
#                     torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
#                 else:
#                     torch.nn.init.constant_(lin.bias, 0.0)
#                     torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
#
#             if weight_norm:
#                 lin = nn.utils.weight_norm(lin)
#
#             setattr(self, "lin" + str(l), lin)
#
#         self.activation = nn.Softplus(beta=100)
#
#         print(f'out dim: {d_hidden}')
#         self.attention = SelfAttention(d_hidden)
#
#     def forward(self, inputs):
#         print(f'inputs shape: {inputs.shape}')
#         inputs = inputs * self.scale
#         if self.embed_fn_fine is not None:
#             inputs = self.embed_fn_fine(inputs)
#
#         x = inputs
#         for l in range(0, self.num_layers - 1):
#             lin = getattr(self, "lin" + str(l))
#
#             if l in self.skip_in:
#                 x = torch.cat([x, inputs], 1) / np.sqrt(2)
#
#             x = lin(x)
#
#             if l < self.num_layers - 2:
#                 x = self.activation(x)
#
#             # attention
#             # Reshape x for attention
#             # x = x.view(x.size(0), -1, 256)
#             x = x.unsqueeze(1)  # shape becomes (batch, 1, feature)
#             print(f'x shape: {x.shape}')
#             x = self.attention(x)
#             x = x.view(x.size(0), -1)
#
#         return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)


    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)