# %%
import deeplay as dl
import torch.nn as nn
import torch


# %%
def get_unet():
    unet = dl.UNet2d(
        in_channels=1,
        out_channels=1,
        channels=[16, 32, 64, 128],
    )
    unet["encoder", "blocks", 0].configure("layer", kernel_size=5)
    unet["encoder", "blocks", 1:].all.style("resnet", stride=2)
    unet["bottleneck", "blocks", 0].first.style("resnet", stride=2)
    unet["bottleneck", "blocks", 1:-1].all.style("resnet")
    unet["bottleneck", "blocks", -1].first.append(
        dl.Layer(
            nn.LazyConvTranspose2d,
            kernel_size=2,
            stride=2,
            padding=0,
            out_channels=128,
        ),
        "upsample",
    )

    unet["decoder", "blocks", :-1].all.normalized(
        nn.InstanceNorm2d, mode="insert", after="activation"
    )
    unet["decoder", "blocks", -1].first.multi(3)
    unet["decoder", "blocks", -1, "blocks", :].all.configure(
        out_channels=32, in_channels=32
    )
    unet["decoder", "blocks", -1, "blocks", -1].all.configure(out_channels=1)
    unet[..., "activation#:-1"].configure(nn.SiLU)
    return unet


def get_patch_discriminator():
    patch_discriminator = dl.ConvolutionalNeuralNetwork(
        in_channels=2,
        hidden_channels=[64, 128, 256, 512],
        out_channels=1,
    )

    patch_discriminator[..., "layer#:-1"].all.configure(
        kernel_size=4, stride=2, padding=0
    )
    patch_discriminator[..., "layer#-1"].all.configure(
        kernel_size=1, stride=1, padding=0
    )
    patch_discriminator[..., "activation#:-1"].configure(
        nn.LeakyReLU, negative_slope=0.2
    )
    patch_discriminator["blocks", 1:-1].all.normalized(
        nn.InstanceNorm2d, mode="insert", after="layer"
    )
    # unet["decoder", "blocks", -1].first.style("resnet")
    return patch_discriminator


# %%
class GaussianInstanceModel(dl.Application):

    λ_pix = 0.32  # pixel-wise (BCE) weight
    λ_adv = 0.05  # adversarial weight
    λ_peak = 0.1  # peak-suppression weight
    σ_pool = 7  # peak kernel (≈ 2·σ + 1)

    def __init__(self):
        super().__init__()
        self.generator = get_unet()
        self.patch_discriminator = get_patch_discriminator()
        self.bce_pix = nn.SmoothL1Loss()

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.generator.parameters(), lr=3e-4, weight_decay=1e-4
        )
        opt_d = torch.optim.AdamW(
            self.patch_discriminator.parameters(), lr=3e-4, weight_decay=1e-4
        )
        return [opt_g, opt_d], []  # no schedulers

    def forward(self, x):
        return self.generator(x)

    @staticmethod
    def peak_loss(logits, k):
        pooled = torch.nn.functional.max_pool2d(logits, k, 1, k // 2)
        diff = logits - pooled
        mask = diff < 0  # negative where not the max
        return torch.mean(torch.abs(logits[mask]))  # L1 on non-max pixels

    # --------------------------------------------------------------------- #
    # training loop                                                         #
    # --------------------------------------------------------------------- #
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch  # y ∈ [0,1] Gaussians (B,1,H,W)

        # ------------------------------------------------------------------ #
        # 0. generator forward                                              #
        # ------------------------------------------------------------------ #
        g_logits = self.generator(x)  # (B,1,H,W) raw logits

        # ------------------------------------------------------------------ #
        # 1. pixel loss (BCE)                                                #
        # ------------------------------------------------------------------ #
        pixel_loss = self.bce_pix(g_logits, y)

        # ------------------------------------------------------------------ #
        # 2. adversarial part                                                #
        # ------------------------------------------------------------------ #

        # concat image for conditional D
        fake_in = torch.cat([x, g_logits.sigmoid()], dim=1).detach()
        real_in = torch.cat([x, y], dim=1)

        if optimizer_idx == 1:  # discriminator step
            # real = 1, fake = 0  (hinge)
            real_logits = self.patch_discriminator(real_in)
            fake_logits = self.patch_discriminator(fake_in)

            loss_d = (
                torch.relu(1.0 - real_logits).mean()
                + torch.relu(1.0 + fake_logits).mean()
            )
            self.log(
                "train/D_loss",
                loss_d,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=y.size(0),
            )
            return loss_d

        # ------------------------------------------------------------------ #
        # 3. generator step (optimizer_idx == 0)                             #
        # ------------------------------------------------------------------ #
        fake_in = torch.cat([x, torch.sigmoid(g_logits)], dim=1)
        adv_logits = self.patch_discriminator(fake_in)
        adv_loss = -adv_logits.mean()

        peak_loss = self.peak_loss(torch.sigmoid(g_logits), self.σ_pool)

        loss_g = (
            self.λ_pix * pixel_loss + self.λ_adv * adv_loss + self.λ_peak * peak_loss
        )

        self.log_dict(
            {
                "train/G_loss": loss_g,
                "train/pixel": pixel_loss,
                "train/adv": adv_loss,
                "train/peak": peak_loss,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=y.size(0),
        )
        return loss_g
