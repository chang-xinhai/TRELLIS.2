"""
Multi-image to 3D generation using TRELLIS 2.

This example demonstrates multi-image conditioning with two fusion modes:
- 'stochastic': Cycles through images at each step (memory efficient)
- 'multidiffusion': Averages all images at each step (higher quality)
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# Configuration
IMAGE_PATHS = [
    "assets/example_image/T.png",
    "assets/example_image/T.png",  # Add more image paths here
]
FUSION_MODE = 'multidiffusion'  # 'stochastic' or 'multidiffusion'
RESOLUTION = '1024'  # '512', '1024', or '1536'

# 1. Setup Environment Map
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Load Multiple Images
images = [Image.open(path) for path in IMAGE_PATHS]

# 4. Run Multi-Image Pipeline
pipeline_type = {'512': '512', '1024': '1024_cascade', '1536': '1536_cascade'}[RESOLUTION]
meshes = pipeline.run_multi_image(
    images,
    seed=42,
    pipeline_type=pipeline_type,
    mode=FUSION_MODE,
)
mesh = meshes[0]
mesh.simplify(16777216)  # nvdiffrast limit

# 5. Render Video
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample_multi.mp4", video, fps=15)

# 6. Export to GLB
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export("sample_multi.glb", extension_webp=True)
