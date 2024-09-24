import numpy as np
import PIL
import torch
from executorch import exir
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from torchtune.models.clip._transform import CLIPImageTransform
from torchtune.models.flamingo._component_builders import (
    flamingo_decoder,
    flamingo_vision_encoder,
)
from torchvision.transforms.v2 import functional as F
from functools import lru_cache, wraps
import time

max_seq_len = 8192
in_channels = 3
tile_size = 448
max_num_tiles = 4
generate_aot_inductor = False

@lru_cache(maxsize=1)
def get_vision_encoder():
    return flamingo_vision_encoder(
        patch_size=14,
        num_heads=16,
        clip_embed_dim=1280,
        clip_num_layers=32,
        clip_hidden_states=[3, 7, 15, 23, 30],
        decoder_embed_dim=4096,
        num_layers_projection=8,
        tile_size=448,
        max_num_tiles=4,
        in_channels=3,
    )


@lru_cache(maxsize=1)
def get_text_decoder():
    return flamingo_decoder(
        vocab_size=128_256,
        num_layers=32,
        fusion_interval=4,
        num_special_tokens=8,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        encoder_max_seq_len=8192,
        rope_base=500000.0,
        intermediate_dim=14336,
    )


@lru_cache(maxsize=1)
def get_flamingo():
    return DeepFusionModel(
        encoder=get_vision_encoder(),
        decoder=get_text_decoder(),
    )


@lru_cache(maxsize=1)
def get_sample_preprocess_outputs():
    image = (np.random.rand(800, 600, 3) * 255).astype(np.uint8)
    image_pil = PIL.Image.fromarray(image)
    image_tensor = F.to_dtype(
        F.grayscale_to_rgb_image(F.to_image(image_pil)), scale=True
    )
    image_transform = CLIPImageTransform(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        tile_size=tile_size,
        possible_resolutions=None,
        max_num_tiles=4,
        resample="bilinear",
        resize_to_max_canvas=True,
    )
    return image_transform(image=image_pil)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
        return result
    return wrapper


@timeit
def run_vision_encoder_eager(vision_encoder, image, aspect_ratio):
    return vision_encoder(image, aspect_ratio)


def benchmark_vision_encoder(vision_encoder, image, aspect_ratio):
    # warm up run
    vision_encoder(image, aspect_ratio)
    # time it
    total_time = 0
    for _ in range(30):
        start_time = time.time()
        res = vision_encoder(image, aspect_ratio)
        total_time += time.time() - start_time
    return total_time / 30, res


def benchmark_all_vision_encoder():
    preprocess_outputs = get_sample_preprocess_outputs()
    image = preprocess_outputs["image"].reshape(
        (
            1,
            1,
            4,
            in_channels,
            tile_size,
            tile_size,
        )
    )
    # Eager
    aspect_ratio = preprocess_outputs["aspect_ratio"].reshape(1, 1, 2)
    print("image shape:", image.shape)
    for dtype in [torch.bfloat16, torch.float32]:
        image = image.to(dtype=dtype)
        aspect_ratio = aspect_ratio.to(dtype=dtype)
        vision_encoder = get_vision_encoder().to(dtype=dtype).eval()
        print(f"-----------------------------------Eager Mode {dtype} CPU-----------------------------------")
        avg, eager_res = benchmark_vision_encoder(vision_encoder, image, aspect_ratio)
        print(f"Averaged time: {avg}")

        # # Torch.compile
        # print(f"-----------------------------------Torch.compile {dtype} CPU-----------------------------------")
        # with torch.no_grad():
        #     compiled_vision_encoder = torch.compile(vision_encoder, mode="reduce-overhead")
        # # warm up run
        # compiled_vision_encoder(image, aspect_ratio)
        # # time it
        # avg, compiled_res = benchmark_vision_encoder(compiled_vision_encoder, image, aspect_ratio)
        # print(f"Averaged time: {avg}")
        # print(f"Close to eager? {torch.allclose(eager_res, compiled_res)}")

        # # torch.export
        # print(f"-----------------------------------Torch.export {dtype} CPU-----------------------------------")
        dim = torch.export.Dim("num_tiles", min=1, max=max_num_tiles)
        image_dynamic_dim = {
            0: 1,
            1: 1,
            2: dim,
            3: 3,
            4: tile_size,
            5: tile_size,
        }
        # ep = torch.export.export(
        #     vision_encoder,
        #     (image, aspect_ratio),
        #     dynamic_shapes=(image_dynamic_dim, None),
        # )
        # avg, exported_res = benchmark_vision_encoder(ep.module(), image, aspect_ratio)
        # print(f"Averaged time: {avg}")
        # print(f"Close to eager? {torch.allclose(eager_res, exported_res)}")

        # AOTInductor
        print(f"-----------------------------------AOTInductor {dtype} CPU-----------------------------------")
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            so = torch._export.aot_compile(
                vision_encoder,
                args=(image, aspect_ratio),
                options={"aot_inductor.output_path": "/tmp/vision_transformer.so"},
                dynamic_shapes=(image_dynamic_dim, None),
            )
        aot_loaded = torch._export.aot_load(so, device="cpu")
        avg, aoti_res = benchmark_vision_encoder(aot_loaded, image, aspect_ratio)
        print(f"Averaged time: {avg}")
        print(f"Close to eager? {torch.allclose(eager_res, aoti_res)}")
        print(f"-----------------------------------Eager Mode {dtype} GPU-----------------------------------")
        image_cuda = image.to(device="cuda")
        aspect_ratio_cuda = aspect_ratio.to(device="cuda")
        vision_encoder_cuda = vision_encoder.to(device="cuda")
        avg, eager_res_cuda = benchmark_vision_encoder(vision_encoder_cuda, image_cuda, aspect_ratio_cuda)
        print(f"Averaged time: {avg}")
        print(f"Close to eager? {torch.allclose(eager_res, eager_res_cuda.cpu())}")
        # Torch.compile
        # print("-----------------------------------Torch.compile fp32 GPU-----------------------------------")
        # with torch.no_grad():
        #     compiled_vision_encoder_cuda = torch.compile(vision_encoder_cuda, mode="reduce-overhead")
        # # warm up run
        # compiled_vision_encoder_cuda(image_cuda, aspect_ratio_cuda)
        # # time it
        # compiled_res = run_vision_encoder_eager(compiled_vision_encoder_cuda, image_cuda, aspect_ratio_cuda)
        # print(f"Close to eager? {torch.allclose(eager_res, compiled_res)}")

        # print("-----------------------------------AOTInductor fp32 GPU-----------------------------------")
        # with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        #     so = torch._export.aot_compile(
        #         vision_encoder_cuda,
        #         args=(image_cuda, aspect_ratio_cuda),
        #         options={"aot_inductor.output_path": "/tmp/vision_transformer.so"},
        #         dynamic_shapes=(image_dynamic_dim, None),
        #     )
        # aoti_res = run_vision_encoder_eager(torch._export.aot_load(so, device="cuda"), image_cuda, aspect_ratio_cuda)


def aoti_export_text_decoder():
    text_decoder = get_text_decoder()
    with torch.no_grad():
        dim = torch.export.Dim("token_dim", min=1, max=max_seq_len)
        dim_enc = torch.export.Dim("enc_dim", min=1, max=2050)

        dynamic_shapes = {
            "tokens": {0: 1, 1: dim},
            "encoder_input": {0: 1, 1: dim_enc, 2: 4096},
            # "encoder_mask": None,#{0:1, 1:dim, 2:dim_enc},
            "input_pos": {0: dim},
        }
        tokens_dynamic_dim = {0: 1, 1: dim}
        encoder_input_dynamic_dim = {0: 1, 1: dim_enc, 2: 4096}
        input_pos_dynamic_dim = {0: dim}

        tokens = torch.ones(1, 64, dtype=torch.int)
        input_pos = torch.ones(64, dtype=torch.int)

        encoder_input = torch.ones(1, 2050, 4096)
        encoder_mask = torch.ones(1, 64, 2050)

        text_decoder.setup_caches(1, torch.float32)
        print("Start to generate aoti for text decoder")
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            so = torch._export.aot_compile(
                text_decoder,
                (tokens,),
                {"encoder_input": encoder_input, "input_pos": input_pos},
                options={"aot_inductor.output_path": "/tmp/text_decoder.so"},
                dynamic_shapes=dynamic_shapes,
            )

def main():
    # get image and aspect ratio input
    aoti_export_text_decoder()
    

if __name__ == "__main__":
    main()
