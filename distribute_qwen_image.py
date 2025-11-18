#!/usr/bin/env python3
"""
Distributed Qwen Image Pipeline

A script for running Qwen-Image models with distributed GPU support.
Supports multi-GPU inference with configurable device placement.
"""

import torch
import time
import copy
import argparse
import os
import sys
from diffsynth.pipelines.qwen_image import QwenImagePipeline

from diffsynth.pipelines.qwen_image import ModelConfig
from typing import List
from tqdm import tqdm
from PIL import Image
import torch.profiler

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Distributed Qwen Image Pipeline")
    
    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the Qwen-Image model directory")

    
    # Device configuration
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated list of GPU devices (e.g., 'cuda:0,cuda:1,cuda:2')")
    parser.add_argument("--num-stages", type=int, default=None,
                        help="Number of transformer stages to distribute (default: auto)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference (default: 1)")
    
    # Data configuration
    parser.add_argument("--input-image", type=str, required=True,
                        help="Path to input image file")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt (default: empty)")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory for generated images (default: ./output)")
    
    # Performance optimization
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 precision (default: True)")
    parser.add_argument("--no-bf16", action="store_false", dest="bf16",
                        help="Disable bfloat16 precision")

    parser.add_argument("--reduce-resolution", action="store_true", default=False,
                        help="Reduce resolution to save memory (default: False)")
    
    # Inference parameters
    parser.add_argument("--height", type=int, default=1328,
                        help="Output image height (default: 1328)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Output image width (default: 1024)")
    parser.add_argument("--num-inference-steps", type=int, default=4,
                        help="Number of denoising steps (default: 4)")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Profiling
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable PyTorch profiling (default: False)")
    parser.add_argument("--profile-dir", type=str, default="./qwen_profile",
                        help="Directory for profiling output (default: ./qwen_profile)")
    
    return parser.parse_args()

def validate_args(args):
    """Validate command line arguments"""
    # Check model path
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")
    
    # Check input image
    if not os.path.exists(args.input_image):
        raise ValueError(f"Input image does not exist: {args.input_image}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate devices
    if args.devices:
        device_list = [d.strip() for d in args.devices.split(',')]
        for device in device_list:
            if not device.startswith('cuda:'):
                raise ValueError(f"Invalid device format: {device}. Must be 'cuda:X' format")
    
    return args

def load_pipe_cpu(model_path, bf16=True, attention_slicing=True):
    """
    Load pipeline to CPU first to avoid OOM during load
    
    Args:
        model_path: Path to the model directory
        bf16: Whether to use bfloat16 precision
        attention_slicing: Whether to enable attention slicing
    """
    # Use consistent dtype for both computation and offload
    target_dtype = torch.bfloat16 if bf16 else torch.float16
    
    model_configs = [
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509",
                    origin_file_pattern=f"{model_path}/transformer/diffusion_pytorch_model*.safetensors",
                    offload_device="cpu", offload_dtype=target_dtype),
        ModelConfig(model_id="Qwen/Qwen-Image",
                    origin_file_pattern=f"{model_path}/text_encoder/model*.safetensors",
                    offload_device="cpu", offload_dtype=target_dtype),
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509",
                    origin_file_pattern=f"{model_path}/vae/diffusion_pytorch_model.safetensors",
                    offload_device="cpu", offload_dtype=target_dtype),
    ]
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=target_dtype,
        device="cpu",
        model_configs=model_configs,
        processor_config=ModelConfig(
            model_id="Qwen/Qwen-Image-Edit-2509",
            origin_file_pattern=f"{model_path}/processor/"
        )
    )
    
    # Enable memory optimization
    if attention_slicing:
        try:
            pipe.enable_attention_slicing("max")
            print("Enabled attention slicing for memory optimization")
        except Exception as e:
            print(f"Failed to enable attention slicing: {e}")
    
    # Enable VRAM management
    try:
        pipe.enable_vram_management()
        print("Enabled VRAM management")
    except Exception as e:
        print(f"Failed to enable VRAM management: {e}")
    
    return pipe

def create_distributed_dit(pipe, device_list, num_stages=None):
    """
    Create a distributed version of the DiT model that works with model_fn_qwen_image
    
    Args:
        pipe: The pipeline object
        device_list: List of GPU devices to use
        num_stages: Number of transformer stages to distribute (auto if None)
    """
    original_dit = pipe.dit
    
    # Auto-determine number of stages if not specified
    if num_stages is None:
        num_stages = max(1, min(len(device_list)-1, 6))
    
    class DistributedDiT(torch.nn.Module):
        def __init__(self, original_dit, placed_stages):
            super().__init__()
            self.original_dit = original_dit
            self.placed_stages = placed_stages
            
            # Copy necessary attributes from original DiT
            self.img_in = original_dit.img_in
            self.time_text_embed = original_dit.time_text_embed
            self.txt_norm = original_dit.txt_norm
            self.txt_in = original_dit.txt_in
            self.pos_embed = original_dit.pos_embed
            self.norm_out = original_dit.norm_out
            self.proj_out = original_dit.proj_out
            self.process_entity_masks = original_dit.process_entity_masks
            
            # Use the first transformer block device for initial components
            # This ensures consistency with where the latents will be processed
            if len(placed_stages) > 1:
                first_transformer_device = placed_stages[1][1]  # First transformer stage
            else:
                first_transformer_device = placed_stages[0][1]
                
            self.img_in.to(first_transformer_device)
            self.time_text_embed.to(first_transformer_device)
            self.txt_norm.to(first_transformer_device)
            self.txt_in.to(first_transformer_device)
            self.pos_embed.to(first_transformer_device)
            
            # Move final components to last transformer device
            if len(placed_stages) > 1:
                last_transformer_device = placed_stages[-2][1]  # Last transformer stage
            else:
                last_transformer_device = placed_stages[0][1]
                
            self.norm_out.to(last_transformer_device)
            self.proj_out.to(last_transformer_device)
            
            # Store device for reference
            self.first_device = first_transformer_device
            self.last_device = last_transformer_device
        
        @property
        def transformer_blocks(self):
            """
            Return a distributed version of transformer blocks
            """
            class DistributedBlocks:
                def __init__(self, placed_stages):
                    self.placed_stages = placed_stages
                
                def __iter__(self):
                    # Return distributed block wrappers
                    for i, (stage, device) in enumerate(self.placed_stages[1:-1]):  # Skip initial and final stages
                        yield DistributedBlock(stage, device, i)
                
                def __len__(self):
                    return len(self.placed_stages) - 2  # Exclude initial and final stages
            
            return DistributedBlocks(self.placed_stages)
    
    class DistributedBlock:
        def __init__(self, stage, device, block_id):
            self.stage = stage
            self.device = device
            self.block_id = block_id
        
        def __call__(self, image, text, temb, image_rotary_emb, attention_mask=None, enable_fp8_attention=False):
            # Move inputs to this block's device
            image = image.to(self.device, non_blocking=True)
            text = text.to(self.device, non_blocking=True)
            temb = temb.to(self.device, non_blocking=True)
            if image_rotary_emb is not None:
                # image_rotary_emb is a tuple (vid_freqs, txt_freqs)
                if isinstance(image_rotary_emb, tuple):
                    image_rotary_emb = tuple(emb.to(self.device, non_blocking=True) for emb in image_rotary_emb)
                else:
                    image_rotary_emb = image_rotary_emb.to(self.device, non_blocking=True)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device, non_blocking=True)
            
            # Forward through this stage
            return self.stage(image, text, temb, image_rotary_emb, attention_mask, enable_fp8_attention)
    
    # Extract stages and create distributed DiT
    stages = split_dit_into_stages(original_dit, num_stages)
    stage_devices = device_list[1:1+num_stages]
    placed_stages = place_stages_on_devices(stages, stage_devices)
    
    return DistributedDiT(original_dit, placed_stages)

class QwenBlockStage(torch.nn.Module):
    """Custom stage wrapper for QwenImageTransformerBlock that handles multiple arguments"""
    def __init__(self, blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
    
    def forward(self, image, text, temb, image_rotary_emb, attention_mask=None, enable_fp8_attention=False):
        for block in self.blocks:
            text, image = block(image, text, temb, image_rotary_emb, attention_mask, enable_fp8_attention)
        return text, image

def split_dit_into_stages(dit, n_stages: int) -> List[torch.nn.Module]:
    """
    Splits the DiT transformer blocks into stages for distribution
    """
    blocks = list(dit.transformer_blocks)
    
    # Initial stage contains preprocessing components (empty for now)
    initial_stage = QwenBlockStage([])
    
    # Split transformer blocks across stages
    k = max(1, len(blocks) // n_stages)
    stages = [initial_stage]
    
    print(f"Split {len(blocks)} blocks into {n_stages} stages")
    i = 0
    for stage_idx in range(n_stages):
        start = i
        end = start + k if stage_idx < n_stages - 1 else len(blocks)
        print(f"Stage {stage_idx}: blocks {start} to {end-1}")
        if start < len(blocks):
            stage_blocks = blocks[start:end]
            stage = QwenBlockStage(stage_blocks)
            stages.append(stage)
        i = end
    
    # Final stage contains postprocessing components (empty for now)
    final_stage = QwenBlockStage([])
    stages.append(final_stage)
    
    return stages

def place_stages_on_devices(stages, devices):
    """
    Place stages on specified devices
    """
    placed = []
    for i, stage in enumerate(stages):
        dev = devices[i % len(devices)]
        stage = stage.to(dev)
        placed.append((stage, dev))
    return placed


def distributed_model_fn_qwen_image(
    dit=None,
    blockwise_controlnet=None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    prompt_emb_mask=None,
    height=None,
    width=None,
    blockwise_controlnet_conditioning=None,
    blockwise_controlnet_inputs=None,
    progress_id=0,
    num_inference_steps=1,
    entity_prompt_emb=None,
    entity_prompt_emb_mask=None,
    entity_masks=None,
    edit_latents=None,
    context_latents=None,
    enable_fp8_attention=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    edit_rope_interpolation=False,
    **kwargs
):
    """
    Distributed version of model_fn_qwen_image that works with DistributedDiT
    """
    from einops import rearrange
    # 强制禁用梯度，避免前向构建/保留计算图导致显存累积
    _prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        # 初始化和预处理阶段
        img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        timestep = timestep / 1000
        
        image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
        image_seq_len = image.shape[1]
    
        # 上下文和编辑图像处理
        if context_latents is not None:
            img_shapes += [(context_latents.shape[0], context_latents.shape[2]//2, context_latents.shape[3]//2)]
            context_image = rearrange(context_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=context_latents.shape[2]//2, W=context_latents.shape[3]//2, P=2, Q=2)
            image = torch.cat([image, context_image], dim=1)
        if edit_latents is not None:
            edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
            img_shapes += [(e.shape[0], e.shape[2]//2, e.shape[3]//2) for e in edit_latents_list]
            edit_image = [rearrange(e, "B C (H P) (W Q) -> B (H W) (C P Q)", H=e.shape[2]//2, W=e.shape[3]//2, P=2, Q=2) for e in edit_latents_list]
            # 确保所有edit_image张量都在dit.first_device上，与主image张量保持一致
            # edit_image = [e.to(DEVICE_LIST[0], non_blocking=True) for e in edit_image]
            edit_image = [e.to(image.device, non_blocking=True) for e in edit_image]
            image = torch.cat([image] + edit_image, dim=1)
    
        # 移动到第一个设备进行初始处理
        image = image.to(dit.first_device, non_blocking=True)
        timestep = timestep.to(dit.first_device, non_blocking=True)
        
        # 图像和条件嵌入
        image = dit.img_in(image)
        conditioning = dit.time_text_embed(timestep, image.dtype)
    
        # 文本和位置嵌入处理
        if entity_prompt_emb is not None:
            text, image_rotary_emb, attention_mask = dit.process_entity_masks(
                latents, prompt_emb, prompt_emb_mask, entity_prompt_emb, entity_prompt_emb_mask,
                entity_masks, height, width, image, img_shapes,
            )
        else:
            # 确保prompt_emb也在正确的设备上
            prompt_emb = prompt_emb.to(dit.first_device, non_blocking=True)
            text = dit.txt_in(dit.txt_norm(prompt_emb))
            if edit_rope_interpolation:
                image_rotary_emb = dit.pos_embed.forward_sampling(img_shapes, txt_seq_lens, device=dit.first_device)
            else:
                image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=dit.first_device)
            attention_mask = None
            
        # ControlNet 预处理
        if blockwise_controlnet_conditioning is not None:
            blockwise_controlnet_conditioning = blockwise_controlnet.preprocess(
                blockwise_controlnet_inputs, blockwise_controlnet_conditioning)
    
        for stage_id, stage_wrapper in enumerate(dit.transformer_blocks):
            target_device = stage_wrapper.device
            # 打印该 stage 包含的原始 block 数量，避免和“原始 block”概念混淆
            num_blocks_in_stage = len(stage_wrapper.stage.blocks) if hasattr(stage_wrapper, "stage") else -1
            # print(f"Process stage {stage_id} ({num_blocks_in_stage} blocks) on device {target_device}")
    
            # 直接传递在 first_device 上的 conditioning 张量
            # DistributedBlock.__call__ 内部的 "temb = temb.to(self.device)" 会自动处理设备转移
            text, image = stage_wrapper(
                image=image,
                text=text,
                temb=conditioning, 
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                enable_fp8_attention=enable_fp8_attention,
            )

            if blockwise_controlnet_conditioning is not None:
                image_slice = image[:, :image_seq_len].clone()
                controlnet_output = blockwise_controlnet.blockwise_forward(
                    image=image_slice, conditionings=blockwise_controlnet_conditioning,
                    controlnet_inputs=blockwise_controlnet_inputs, block_id=stage_id,
                    progress_id=progress_id, num_inference_steps=num_inference_steps,
                )
                image[:, :image_seq_len] = image_slice + controlnet_output
        
        # 移动到最后设备进行输出处理
        image = image.to(dit.last_device, non_blocking=True)
        conditioning = conditioning.to(dit.last_device, non_blocking=True)
        
        # 输出处理
        image = dit.norm_out(image, conditioning)
        image = dit.proj_out(image)
        image = image[:, :image_seq_len]
        
        latents = rearrange(image, "B (H W) (C P Q) -> B C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2)
        
        return latents
    finally:
        torch.set_grad_enabled(_prev_grad)

from diffsynth.models.utils import load_state_dict



def build_distributed_pipeline(args, device_list):
    """
    Build a distributed pipeline with the given configuration
    
    Args:
        args: Command line arguments
        device_list: List of GPU devices to use
    """
    pipe = load_pipe_cpu(args.model_path, args.bf16)

    # Put TextEncoder & VAE to dedicated GPUs
    if len(device_list) >= 2:
        text_dev = device_list[0]
        vae_dev = device_list[-1]
    else:
        text_dev = device_list[0] if device_list else "cuda:0"
        vae_dev = device_list[0] if device_list else "cuda:0"

    # Disable VRAM management to ensure consistent device placement
    if hasattr(pipe, 'vram_management_enabled') and pipe.vram_management_enabled:
        pipe.vram_management_enabled = False
        print("Disabled VRAM management to ensure consistent device placement")
    
    # Force text_encoder and all its submodules to text_dev
    pipe.text_encoder.to(text_dev)
    
    # Ensure all parameters are on the correct device
    for p in pipe.text_encoder.parameters():
        p.data = p.data.to(text_dev)
    
    # Ensure all submodules are on the correct device
    for module in pipe.text_encoder.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data = module.weight.data.to(text_dev)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data = module.bias.data.to(text_dev)
    
    # Ensure embed_tokens is on the correct device
    if hasattr(pipe.text_encoder, 'model') and hasattr(pipe.text_encoder.model, 'embed_tokens'):
        pipe.text_encoder.model.embed_tokens.to(text_dev)
        if hasattr(pipe.text_encoder.model.embed_tokens, 'weight'):
            pipe.text_encoder.model.embed_tokens.weight.data = pipe.text_encoder.model.embed_tokens.weight.data.to(text_dev)
    
    pipe.vae.to(vae_dev)

    # Create distributed DiT
    pipe.dit = create_distributed_dit(pipe, device_list, args.num_stages)
    
    # Store device assignments
    pipe._text_dev = text_dev
    pipe._vae_dev = vae_dev
    
    # Set pipe.device to text encoder device to ensure tokenizer outputs are on correct device
    pipe.device = text_dev
    
    # Replace model_fn with distributed version
    pipe.model_fn = distributed_model_fn_qwen_image
    


    # Inference mode: disable training behavior and gradient-related paths
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.dit.eval()
    
    # Ensure all model parameters are in the correct dtype
    target_dtype = torch.bfloat16 if args.bf16 else torch.float16
    pipe.text_encoder = pipe.text_encoder.to(dtype=target_dtype)
    pipe.vae = pipe.vae.to(dtype=target_dtype)
    
    # Ensure pipe's dtype setting is consistent
    pipe.torch_dtype = target_dtype
       
    # Print device allocation information
    print(f"Device allocation:")
    print(f"  Text Encoder: {text_dev}")
    print(f"  VAE: {vae_dev}")
    print(f"  DiT first device: {pipe.dit.first_device}")
    print(f"  DiT last device: {pipe.dit.last_device}")
    print(f"  Total stages: {len(pipe.dit.placed_stages)}")

    return pipe

def main():
    """Main function to run the distributed Qwen image pipeline"""
    try:
        # Parse and validate arguments
        args = parse_args()
        args = validate_args(args)
        
        # Determine device list
        if args.devices:
            device_list = [d.strip() for d in args.devices.split(',')]
        else:
            # Auto-detect available GPUs
            num_gpus = torch.cuda.device_count()
            device_list = [f"cuda:{i}" for i in range(num_gpus)]
            if not device_list:
                device_list = ["cuda:0"]
        
        print(f"Using devices: {device_list}")
        
        # Build distributed pipeline
        pipe = build_distributed_pipeline(args, device_list)
        
        # Ensure TextEncoder and VAE weights are fixed on assigned devices
        pipe.text_encoder.to(pipe._text_dev)
        pipe.vae.to(pipe._vae_dev)

        # Load input image
        img = [Image.open(args.input_image)]
        
        # Prepare inputs following the QwenImagePipeline.__call__ pattern
        inputs_shared = {
            "cfg_scale": args.cfg_scale,
            "input_image": None,
            "denoising_strength": 1.0,
            "inpaint_mask": None,
            "inpaint_blur_size": None,
            "inpaint_blur_sigma": None,
            "height": 768 if args.reduce_resolution else args.height,
            "width": 768 if args.reduce_resolution else args.width,
            "seed": args.seed,
            "rand_device": pipe._text_dev,
            "enable_fp8_attention": False,
            "num_inference_steps": args.num_inference_steps,
            "blockwise_controlnet_inputs": None,
            "tiled": False,
            "tile_size": 128,
            "tile_stride": 64,
            "eligen_entity_prompts": None,
            "eligen_entity_masks": None,
            "eligen_enable_on_negative": False,
            "edit_image": img,
            "edit_image_auto_resize": True,
            "edit_rope_interpolation": False,
            "context_image": None,
        }
        inputs_posi = {"prompt": args.prompt}
        inputs_nega = {"negative_prompt": args.negative_prompt}
        
        print(f"Finished loading pipeline")
        
        # Set scheduler timesteps
        pipe.scheduler.set_timesteps(
            inputs_shared["num_inference_steps"], 
            denoising_strength=inputs_shared["denoising_strength"], 
            dynamic_shift_len=(inputs_shared["height"] // 16) * (inputs_shared["width"] // 16)
        )
        
        print(f"Finished setting scheduler timesteps")
        
        # Process through all pipeline units to prepare inputs
        for unit in pipe.units:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
        
        target_device = pipe.dit.first_device
        inputs_shared["latents"] = inputs_shared["latents"].to(target_device)
        print(f"Moved initial latents to {target_device}")

        # Pre-move constant prompt embeddings
        inputs_posi["prompt_emb"] = inputs_posi["prompt_emb"].to(pipe.dit.first_device)
        if "prompt_emb" in inputs_nega:
            inputs_nega["prompt_emb"] = inputs_nega["prompt_emb"].to(pipe.dit.first_device)
        print(f"Pre-moved prompt embeddings to {pipe.dit.first_device}")

        # Load models to their assigned devices
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        
        print(f"Finished loading models to their assigned devices")
        
        # Setup profiling if enabled
        if args.profile:
            profiler_schedule = torch.profiler.schedule(
                wait=1,      # Wait 1 step for model warmup
                warmup=1,    # Warmup 1 step to skip JIT compilation overhead
                active=3,    # Capture next 3 steps (sufficient samples)
                repeat=1     # Repeat once
            )
            
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=profiler_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
        else:
            profiler = None

        # Denoising loop
        if profiler:
            profiler.__enter__()
        
        try:
            for progress_id, timestep in tqdm(enumerate(pipe.scheduler.timesteps), total=len(pipe.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.dit.first_device)
                
                # Positive prediction
                with torch.inference_mode():
                    noise_pred_posi = pipe.model_fn(
                        **models, 
                        **inputs_shared, 
                        **inputs_posi, 
                        timestep=timestep, 
                        progress_id=progress_id
                    )
                
                # Negative prediction for CFG
                if inputs_shared["cfg_scale"] != 1.0:
                    with torch.inference_mode():
                        noise_pred_nega = pipe.model_fn(
                            **models, 
                            **inputs_shared, 
                            **inputs_nega, 
                            timestep=timestep, 
                            progress_id=progress_id
                        )
                    noise_pred = noise_pred_nega + inputs_shared["cfg_scale"] * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi
                
                # Ensure noise_pred and latents are on the same device
                latents_device = inputs_shared["latents"].device
                noise_pred = noise_pred.to(latents_device, non_blocking=True)
                timestep_for_scheduler = timestep.squeeze(0).to(latents_device, non_blocking=True)
                
                # Scheduler step
                inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

                if profiler:
                    profiler.step()

                if args.profile and progress_id >= 5:  # wait + warmup + active
                    break
        
        finally:
            if profiler:
                profiler.__exit__(None, None, None)
                # Print profiling results
                print("-" * 80)
                print("PyTorch Profiler CLI Report:")
                print(profiler.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))
                print("-" * 80)

        # Clean up conditioning cache after inference
        if hasattr(pipe.dit, 'conditioning_cache'):
            del pipe.dit.conditioning_cache
            torch.cuda.empty_cache()
        
        # Decode using VAE
        print(f"Finished denoising loop")
        pipe.vae.to(pipe._vae_dev)
        
        # Move latents to VAE device to avoid cross-device convolution errors
        with torch.inference_mode():
            latents_for_vae = inputs_shared["latents"].to(pipe._vae_dev, non_blocking=True)
            image = pipe.vae.decode(latents_for_vae, device=pipe._vae_dev, 
                                  tiled=inputs_shared.get("tiled", False), 
                                  tile_size=inputs_shared.get("tile_size", 128), 
                                  tile_stride=inputs_shared.get("tile_stride", 64))
        
        # Convert to PIL image and save
        image = pipe.vae_output_to_image(image)
        
        # Save output image
        output_filename = f"output_{int(time.time())}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        image.save(output_path)
        print(f"Saved output image to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
    print("Saved output.png")
