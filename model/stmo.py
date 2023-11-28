import torch
import torch.nn as nn
from model.block.vanilla_transformer_encoder import Transformer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.25):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        #self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w1 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        #self.w2 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class FCBlock(nn.Module):

    def __init__(self, channel_in, channel_out, linear_size, block_num):
        super(FCBlock, self).__init__()

        self.linear_size = linear_size
        self.block_num = block_num
        self.layers = []
        self.channel_in = channel_in
        self.stage_num = 3
        self.p_dropout = 0.1
        #self.fc_1 = nn.Linear(self.channel_in, self.linear_size)
        self.fc_1 = nn.Conv1d(self.channel_in, self.linear_size, kernel_size=1)
        self.bn_1 = nn.BatchNorm1d(self.linear_size)
        for i in range(block_num):
            self.layers.append(Linear(self.linear_size, self.p_dropout))
        #self.fc_2 = nn.Linear(self.linear_size, channel_out)
        self.fc_2 = nn.Conv1d(self.linear_size, channel_out, kernel_size=1)

        self.layers = nn.ModuleList(self.layers)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):

        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        for i in range(self.block_num):
            x = self.layers[i](x)
        x = self.fc_2(x)

        return x

class SynthAttention(nn.Module):
    def __init__(self, in_dims, out_dims, reduce_factor=1):  # L, emb
        super().__init__()

        reduced_dims = out_dims // reduce_factor
        self.dense = nn.Linear(in_dims, reduced_dims)
        self.reduce_factor = reduce_factor
        self.se = SELayer(out_dims, r=4)
        self.value_reduce = nn.Linear(out_dims, reduced_dims)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        energy = self.dense(x)  # b, channel, reduced_dim
        energy = self.se(energy)

        attention = self.softmax(energy)

        value = x
        if self.reduce_factor > 1:
            value = self.value_reduce(value.transpose(1, 2)).transpose(1, 2)  # b, reduced_dim, t

        out = torch.bmm(attention, value)
        return out


class SynthMix(nn.Module):
    def __init__(self, temporal_dim, channel_dim, proj_drop, reduce_token, reduce_channel):  # L, emb
        super().__init__()
        # generate temporal matrix
        self.synth_token = SynthAttention(channel_dim, temporal_dim, reduce_token)  # reduce factor for window # l, emb -> l, l * l, emb
        # generate spatial matrix
        self.synth_channel = SynthAttention(temporal_dim, channel_dim, reduce_channel)  # reduce factor for embedding # emb, l -> emb, emb//r * emb//r, l

        self.reweight = Mlp(temporal_dim, temporal_dim // 4, temporal_dim * 2)
        self.gap=nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Linear(channel_dim, channel_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # b, L, emb
        B, T, C = x.shape

        t = self.synth_token(x)  # b, t, c  torch.Size([160, 243, 128])
        c = self.synth_channel(x.transpose(1, 2)).transpose(1, 2)  # b, c, t

        # re-weight
        t = t.transpose(1, 2)
        c = c.transpose(1, 2)
        a = self.gap((t + c).transpose(1, 2)).squeeze(-1)  # shape: batch, emb
        a = self.reweight(a).reshape(B, T, 2).permute(2, 0, 1).softmax(dim=0) # 2, batch, channel
        s = torch.einsum("nble, nbe -> nble", [torch.stack([t, c], 0), a])
        x = torch.sum(s, dim=0)
        x = x.transpose(1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, r=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # batch, channel
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SynthEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout, temporal_dim, reduce_token, reduce_channel):
        super().__init__()
        self.synth_att = SynthMix(temporal_dim=temporal_dim, channel_dim=d_model, proj_drop=dropout,
                                        reduce_token=reduce_token, reduce_channel=reduce_channel)
        self.ff = Mlp(in_features=d_model, hidden_features=d_model * expansion_factor, out_features=d_model,
                      drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.synth_att(x))
        x = self.norm2(x + self.ff(x))
        return x



class SynthEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout, temporal_dim, reduce_token, reduce_channel):
        super().__init__()
        self.synth_att = SynthMix(temporal_dim=temporal_dim, channel_dim=d_model, proj_drop=dropout,
                                        reduce_token=reduce_token, reduce_channel=reduce_channel)
        self.ff = Mlp(in_features=d_model, hidden_features=d_model * expansion_factor, out_features=d_model,
                      drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.synth_att(x))
        x = self.norm2(x + self.ff(x))
        return x

class SynthEncoder(nn.TransformerEncoder):
    def __init__(self, d_model=128, expansion_factor=2, dropout=0.1, num_layers=5, window_size=11, reduce_token=1,
                 reduce_channel=1):
        encoder_layer = SynthEncoderLayer(d_model=d_model, expansion_factor=expansion_factor, dropout=dropout,
                                          temporal_dim=window_size, reduce_token=reduce_token,
                                          reduce_channel=reduce_channel)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        stride_num = args.stride_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.encoder = FCBlock(2*self.num_joints_in, channel, 2*channel, 1)

        self.encoderMIX = SynthEncoder(d_model=256,
                                    expansion_factor=2,
                                    dropout=0.1,
                                    num_layers=5,
                                    window_size=243,
                                    reduce_token=4,
                                    reduce_channel=8)
        self.encoder_emb = nn.Linear(34, 256)

        self.Transformer = Transformer(layers, channel, d_hid, length=length)
        self.Transformer_reduce = Transformer_reduce(len(stride_num), channel, d_hid, \
            length=length, stride_num=stride_num)
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        )

    def forward(self, x): # torch.Size([160, 2, 243, 17, 1])
        x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous() # torch.Size([160, 243, 17, 2])
        x_shape = x.shape

        x = x.view(x.shape[0], x.shape[1], -1) # torch.Size([160, 243, 34])
        


        ## mixattention
        x = self.encoder_emb(x)
        x = self.encoderMIX(x)





        # x = x.permute(0, 2, 1).contiguous() # torch.Size([160, 34, 243]) 交换 23 维度

        # x = self.encoder(x) # 空间依赖 torch.Size([160, 256, 243]) 操作 34->256

        # x = x.permute(0, 2, 1).contiguous() # torch.Size([160, 243, 256])
        # x = self.Transformer(x)  # torch.Size([160, 243, 256])

        x_VTE = x
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.fcn_1(x_VTE) 

        x_VTE = x_VTE.view(x_shape[0], self.num_joints_out, -1, x_VTE.shape[2])
        x_VTE = x_VTE.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # torch.Size([160, 3, 243, 17, 1]) 全帧

        x = self.Transformer_reduce(x) 
        x = x.permute(0, 2, 1).contiguous() 
        x = self.fcn(x) 

        x = x.view(x_shape[0], self.num_joints_out, -1, x.shape[2])
        x = x.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # torch.Size([160, 3, 1, 17, 1]) 单帧
        
        return x, x_VTE




