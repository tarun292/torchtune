from typing import List, Optional, Callable

import torch
from torchtune.modules import VisionTransformer, CLSProjection
from torchtune.models.clip._position_embeddings import TokenPositionalEmbedding, TiledTokenPositionalEmbedding, TilePositionalEmbedding

import logging

logger = logging.getLogger(__name__)

def clip_vision_encoder(
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    act_layer: Callable,
    indices_return_hidden: Optional[List[int]] = None,
    output_cls_projection: bool = False,
    tile_size: int = 512,
    patch_size: int = 14,
    max_num_tiles: int = 4,
    mlp_ratio: float = 4.0,
    in_channels: int = 3,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    cls_output_dim: int = 512,
) -> VisionTransformer:

    logger.info("Instantiating clip model...")

    patch_grid_size = tile_size // patch_size

    cls_projection = CLSProjection(inpt_dim=embed_dim, cls_output_dim=cls_output_dim) if output_cls_projection else None
    
    transformer_layer = torch.nn.TransformerEncoderLayer(
        d_model=embed_dim, 
        nhead=num_heads, 
        dim_feedforward=int(mlp_ratio * embed_dim), 
        dropout=attn_dropout, 
        activation=act_layer, 
        layer_norm_eps=norm_eps, 
        batch_first=True, 
        norm_first=True, 
        bias=True)

    # position embeddings
    if max_num_tiles == 1:
        logger.info("Found max_num_tiles=1. Setting tile_pos_embed to None and using only token_pos_embedding.")
        pre_tile_pos_embed = None
        post_tile_pos_embed = None
        token_pos_embedding = TokenPositionalEmbedding(
            embed_dim=embed_dim, 
            patch_grid_size=patch_grid_size)
    else:
        logger.info(f"Found {max_num_tiles=}. Instantiating tile_pos_embedding and token_pos_embedding.")
        pre_tile_pos_embed = TilePositionalEmbedding(max_num_tiles=max_num_tiles, embed_dim=embed_dim)
        post_tile_pos_embed = TilePositionalEmbedding(max_num_tiles=max_num_tiles, embed_dim=embed_dim)
        token_pos_embedding = TiledTokenPositionalEmbedding(
            max_num_tiles=max_num_tiles, 
            embed_dim=embed_dim, 
            patch_grid_size=patch_grid_size)

    return VisionTransformer(
        patch_grid_size=patch_grid_size,
        num_layers=num_layers,
        layer=transformer_layer,
        token_pos_embedding=token_pos_embedding,
        pre_tile_pos_embed=pre_tile_pos_embed,
        post_tile_pos_embed=post_tile_pos_embed,
        cls_projection=cls_projection,
        indices_return_hidden=indices_return_hidden,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
    )
