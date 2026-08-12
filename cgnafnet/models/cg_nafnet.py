"""Full CG-NAFNet encoder-decoder assembly (design doc §4 diagram).

Canonical NAFNet topology (intro → encoder groups with stride-2 downs →
bottleneck NAFBlocks → mirroring ups with skip add → ending + global residual),
with a cluster-routing chain injected after the NAFBlocks of every stage:

  stage = [NAFBlock(s) -> PCGRMLite -> DegradationPrompt -> FiLM]

Every stage owns its own PCGRM / prompt / FiLM instances (separate prototype
bank per stage, uniform counts). Routing overhead is pooled-vector ops only.

The aux order head is optional (config flag). Disabling it must have zero
effect on the restoration output path: we simply don't call it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aux_order_head import AuxOrderHead
from .degradation_prompt import DegradationPrompt
from .film import FiLM
from .nafnet_blocks import NAFBlock
from .pcgrm_lite import PCGRMLite


class _Stage(nn.Module):
    """NAFBlocks + cluster routing chain for one encoder/decoder stage."""

    def __init__(self, width, num_blocks, num_prototypes, prompt_dim, proj_dim):
        super().__init__()
        self.blocks = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.pcgrm = PCGRMLite(width, proj_dim, num_prototypes)
        self.prompt = DegradationPrompt(num_prototypes, prompt_dim)
        self.film = FiLM(prompt_dim, width)

    def forward(self, x, alpha_store=None):
        x = self.blocks(x)
        alpha, x_hat = self.pcgrm(x)
        if alpha_store is not None:
            alpha_store.append(alpha)
        prompt = self.prompt(alpha)
        return self.film(x, prompt), x_hat


class CGNAFNet(nn.Module):
    """Cluster-Guided Dynamic NAFNet.

    Args:
        img_channel: input/output channels (1 = grayscale).
        width: backbone base width.
        num_stages: encoder/decoder depth.
        blocks_per_stage: NAFBlocks per encoder stage (decoder mirrors).
        num_prototypes_per_stage: prototype count per stage.
        prompt_dim: degradation-prompt dimensionality.
        proj_dim: PCGRM projection dimensionality.
        middle_blk_num: NAFBlocks in the bottleneck.
        aux_order_head: enable the aux order classifier (train only).
    """

    def __init__(
        self,
        img_channel=1,
        width=32,
        num_stages=4,
        blocks_per_stage=(2, 2, 4, 2),
        num_prototypes_per_stage=(3, 3, 3, 3),
        prompt_dim=64,
        proj_dim=64,
        middle_blk_num=1,
        aux_order_head=True,
    ):
        super().__init__()
        assert len(blocks_per_stage) == num_stages, "blocks_per_stage must match num_stages"
        assert len(num_prototypes_per_stage) == num_stages

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)

        # Channel width after each encoder down; bottleneck at width*2^num_stages.
        self.chan_widths = [width * (2 ** i) for i in range(num_stages + 1)]
        self.bottleneck_width = self.chan_widths[-1]

        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for i in range(num_stages):
            self.enc_stages.append(
                _Stage(chan, blocks_per_stage[i],
                       num_prototypes_per_stage[i], prompt_dim, proj_dim)
            )
            # encoder skip is captured at this width; down feeds the next stage
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        assert chan == self.bottleneck_width

        self.middle_blks = nn.Sequential(*[NAFBlock(chan)
                                           for _ in range(middle_blk_num)])

        self.ups = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(num_stages):
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan = chan // 2
            self.dec_stages.append(
                _Stage(chan, blocks_per_stage[num_stages - 1 - i],
                       num_prototypes_per_stage[num_stages - 1 - i], prompt_dim, proj_dim)
            )

        self.aux_head = AuxOrderHead(proj_dim, num_orders=6) if aux_order_head else None
        self.aux_order_head = aux_order_head
        self.padder_size = 2 ** num_stages

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def forward(self, x, return_aux=False, return_cluster_posteriors=False):
        """Restore a degraded image.

        Args:
            x: (B, img_channel, H, W) float [0,1].
            return_aux: return (out, aux_logits) when aux head is enabled.
            return_cluster_posteriors: also return list of alphas per stage.

        Returns (depending on flags):
            - (B, img_channel, H, W)
            - (out, aux_logits)
            - (out, alphas)
            - (out, aux_logits, alphas)
        """
        _, _, h, w = x.shape
        inp = self.check_image_size(x)
        x = self.intro(inp)

        alphas = []
        enc_outs = []
        for i, (stage, down) in enumerate(zip(self.enc_stages, self.downs)):
            x, _ = stage(x, alpha_store=alphas if return_cluster_posteriors else None)
            enc_outs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        deepest_x_hat = None
        for i, (up, stage) in enumerate(zip(self.ups, self.dec_stages)):
            x = up(x)
            skip = enc_outs[::-1][i]
            x = x + skip[:, :, : x.shape[2], : x.shape[3]]
            x, x_hat = stage(x, alpha_store=alphas if return_cluster_posteriors else None)
            deepest_x_hat = x_hat

        out = self.ending(x) + inp
        out = out[:, :, :h, :w]

        if return_aux and self.aux_head is not None and deepest_x_hat is not None:
            logits = self.aux_head(deepest_x_hat)
            if return_cluster_posteriors:
                return out, logits, alphas
            return out, logits
        if return_cluster_posteriors:
            return out, alphas
        return out