import torch
import torch.nn as nn
import math

# Define the U-Net architecture
# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
#         super().__init__()

#         half_dim = time_emb_dims // 2

#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
#         ts = torch.arange(total_time_steps, dtype=torch.float32)
#         emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

#         self.time_blocks = nn.Sequential(
#             nn.Embedding.from_pretrained(emb),
#             nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
#             nn.SiLU(),
#             nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
#         )

#     def forward(self, time):
#         return self.time_blocks(time)


class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels

        # self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # h = self.group_norm(x)
        h = self.BN_1_5(x)
        h = h.reshape(B, self.channels, -1).swapaxes(1,2) # [B, D, C, H, W] --> [B, self.channels,]
        h, _ = self.mhsa(h, h, h)  
        h = h.reshape(B, C, D, H, W)
        return x + h
    
    def BN_1_4(self,x):
        pre_x = x.permute(1,0,2,3)
        pre_x = torch.flatten(pre_x,1,-1)
        mean = pre_x.mean(dim=1)
        std = pre_x.std(dim=1)
        x = (x-mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))/(std.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)+1e-5)
        return x
    
    def BN_1_5(self,x):
        pre_x = x.permute(1,0,2,3,4)
        pre_x = torch.flatten(pre_x,1,-1)
        mean = pre_x.mean(dim=1)
        std = pre_x.std(dim=1)
        x = (x-mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0))/(std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)+1e-5)
        return x
    
    def BN_1_3(self,x):
        pre_x = x.permute(1,0,2)
        pre_x = torch.flatten(pre_x,1,-1)
        mean = pre_x.mean(dim=1)
        std = pre_x.std(dim=1)
        x = (x-mean.unsqueeze(-1).unsqueeze(0))/(std.unsqueeze(-1).unsqueeze(0)+1e-5)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout_rate=0.2, time_emb_dims=256, apply_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act_fn = nn.ReLU()
        # Group 1
        self.normlize_1 = nn.BatchNorm3d(self.in_channels)
        self.conv_1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")
        # Group 2 time embedding
        # self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels)

        # Group 3
        self.normlize_2 = nn.BatchNorm3d(self.out_channels)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.conv_2 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        else:
            self.match_input = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        # print("x0",x.shape)
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)

        # group 2
        # add in timestep embedding
        # h += self.dense_1(self.act_fn(t))[:, :, None, None]

        # group 3
        h = self.act_fn(self.normlize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        # Residual and attention
        h = h + self.match_input(x)
        h = self.attention(h)

        return h

class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)


class UNet3d(nn.Module):
    def __init__(
        self,
        input_channels=24,
        output_channels=1,
        num_res_blocks=2,
        base_channels=24,
        base_channels_multiples=(1, 2, 4),  
        apply_attention=(False, False, True, False),
        dropout_rate=0.3,
        time_multiple=4,
    ):
        super(UNet3d,self).__init__()

        time_emb_dims_exp = base_channels * time_multiple
        self.first = nn.Conv3d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1, padding="same")

        num_resolutions = len(base_channels_multiples)

        # Encoder part of the UNet. Dimension reduction.
        self.encoder_blocks = nn.ModuleList()
        curr_channels = [base_channels]
        in_channels = base_channels

        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks):

                block = ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )
                self.encoder_blocks.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)

            if level != (num_resolutions - 1):
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        # Bottleneck in between
        self.bottleneck_blocks = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=True,
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=False,
                ),
            )
        )

        # Decoder part of the UNet. Dimension restoration with skip-connections.
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_resolutions)): # 3
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks + 1): # 2+1
                encoder_in_channels = curr_channels.pop()
                block = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )
                in_channels = out_channels
                self.decoder_blocks.append(block)

            if level != 0:
                self.decoder_blocks.append(UpSample(in_channels))

        self.final = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=output_channels, kernel_size=3, stride=1, padding="same"),
        )

    def forward(self, x): #t

        # x = x.permute(0,2,1,3,4)
        h = self.first(x)
        outs = [h]

        for layer in self.encoder_blocks:
            h = layer(h)
            outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h)


        
        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()

                h = torch.cat([h, out], dim=1)

            h = layer(h)


        h = self.final(h)
        # h = h.view(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4])

        return h