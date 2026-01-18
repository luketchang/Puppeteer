# Puppeteer Rigging & Animation Guide

Complete walkthrough for rigging and animating 3D models using Puppeteer, including all setup, troubleshooting, and common issues.

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Preparing Your 3D Model](#preparing-your-3d-model)
3. [Preparing Your Video](#preparing-your-video)
4. [Running Rigging](#running-rigging)
5. [Running Animation](#running-animation)
6. [Exporting Results](#exporting-results)
7. [Troubleshooting](#troubleshooting)

---

## Initial Setup

### 1. Clone Repository

```bash
cd /workspace
git clone --recursive git@github.com:yourusername/Puppeteer.git
cd Puppeteer
```

**Important**: Use `--recursive` flag to properly initialize git submodules!

### 2. Install Dependencies

```bash
./install.sh
```

This will:

- Install system dependencies (vim, nano, ffmpeg, etc.)
- Setup conda environment
- Install Python packages
- Download model checkpoints
- Setup third-party dependencies

### 3. Activate Environment

Every time you start a new terminal session:

```bash
conda activate puppeteer
```

**Optional**: Auto-activate on login:

```bash
echo "conda activate puppeteer" >> ~/.bashrc
```

### 4. Setup SSH Keys (for Git)

If you need to push/pull from GitHub:

```bash
./setup_ssh.sh
```

Then add the displayed public key to https://github.com/settings/keys

For persistence across pod restarts (RunPod):

```bash
mkdir -p /workspace/ssh_keys
cp ~/.ssh/id_ed25519* /workspace/ssh_keys/
```

---

## Preparing Your 3D Model

### Required Format

- **File type**: `.obj` (Wavefront OBJ)
- **Location**: `examples/{model_name}.obj`
- **Textures**: Optional (`.mtl` and texture images)

### Example Structure

```
examples/
├── character.obj           # Your 3D model
└── character/
    └── input.mp4          # Animation video (to be added later)
```

### If You Have Textures

If your model has materials/textures, they should be:

- `{model_name}/objs/material.mtl`
- `{model_name}/objs/texture*.png` (PBR textures optional)

**Note**: Textures are optional - the pipeline works with untextured models.

---

## Preparing Your Video

### Video Requirements

**Critical**: Video MUST be square (720x720 recommended)

### Check Video Dimensions

```bash
ffmpeg -i your_video.mp4 2>&1 | grep "Stream.*Video"
```

### Convert to 720x720

#### If Portrait/Landscape (needs scaling):

```bash
ffmpeg -i input_video.mp4 \
  -vf "scale=720:720:force_original_aspect_ratio=decrease,pad=720:720:(ow-iw)/2:(oh-ih)/2" \
  -c:a copy output_720x720.mp4
```

#### If Already Square (just resize):

```bash
ffmpeg -i input_video.mp4 -vf "scale=720:720" output_720x720.mp4
```

### Place Video in Correct Location

```bash
mkdir -p examples/character/
cp output_720x720.mp4 examples/character/input.mp4
```

**Important**: The video file MUST be named `input.mp4`

---

## Running Rigging

Rigging generates the skeleton and skinning weights for your 3D model.

### 1. Prepare Your Model

Ensure only the models you want to rig are in the `examples/` folder:

```bash
# Optional: Move other models out temporarily
mkdir -p temp_objs
mv examples/deer.obj examples/spiderman.obj examples/charizard.obj temp_objs/
```

### 2. Run Rigging Pipeline

```bash
cd /workspace/Puppeteer
conda activate puppeteer
PYOPENGL_PLATFORM=egl ./demo_rigging.sh
```

**Time**: ~5-10 minutes depending on mesh complexity

### 3. Verify Results

Rigging outputs are saved to:

- `results/final_rigging/character.txt` - Final rig file
- `results/skel_results/character_skel.obj` - Skeleton visualization
- `results/skel_results/render_results/character_pred/*.png` - Rendered previews

### 4. View Skeleton Visualization

Download the visualization images:

```bash
# On your local machine
ssh user@your-vm "cat /workspace/Puppeteer/results/skel_results/render_results/character_pred/character_pred_view1.png" > skeleton_view1.png
```

Or use RunPod's file browser to download from:
`/workspace/Puppeteer/results/skel_results/render_results/character_pred/`

---

## Running Animation

Animates your rigged model using video guidance.

### 1. Ensure Video is Prepared

```bash
# Check video exists and is 720x720
ls -lh examples/character/input.mp4
ffmpeg -i examples/character/input.mp4 2>&1 | grep "Stream.*Video"
```

### 2. Configure Animation Settings

Edit `demo_animation.sh` if needed:

- Line 57-58: Set `--seq_name 'character'` (your model name)
- Line 57: Set `--img_size 720` (must match video dimensions)
- Line 57: Adjust `--iter 50` (more iterations = better quality, longer time)

### 3. Run Animation Pipeline

```bash
cd /workspace/Puppeteer
conda activate puppeteer
PYOPENGL_PLATFORM=egl ./demo_animation.sh
```

**Time**: ~30-60 minutes for 50 iterations

### 4. Monitor Progress

The script will:

1. Copy rig and mesh to `examples/character/objs/`
2. Extract video frames
3. Calculate optical flow
4. Track joints and vertices
5. Run optimization (shows progress bar)

---

## Exporting Results

### Animation Videos

Results are saved to: `results/animation/character/character_demo/`

**Main outputs**:

- `concat_output.mp4` - Single view animation
- `concat_output_4view.mp4` - Four camera angles
- `joint.mp4` - Joint tracking visualization
- `point.mp4` - Vertex tracking visualization

### Download Results

#### Method 1: RunPod File Browser (Easiest)

Navigate to the results folder and download through the web interface.

#### Method 2: HTTP Server

```bash
# On VM
cd /workspace/Puppeteer/results/animation/character/character_demo/
python3 -m http.server 8000

# On local machine (new terminal)
ssh -L 8000:localhost:8000 user@your-vm -N

# Open browser to: http://localhost:8000
```

#### Method 3: Tar + SSH

```bash
# On VM
cd /workspace/Puppeteer/results/animation/character
tar czf character_results.tar.gz character_demo/

# On local machine
ssh user@your-vm "cat /workspace/Puppeteer/results/animation/character/character_results.tar.gz" > results.tar.gz
tar xzf results.tar.gz
```

### Export to FBX (For Blender/Unity/Unreal)

**Note**: This requires Blender and can be tricky. Skip if you just want videos.

```bash
# Install Blender to workspace (persists across restarts)
cd /workspace
wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz
tar -xf blender-4.2.0-linux-x64.tar.xz
rm blender-4.2.0-linux-x64.tar.xz

# Install dependencies in Blender's Python
/workspace/blender-4.2.0-linux-x64/4.2/python/bin/python3.11 -m pip install trimesh Pillow numpy scipy

# Export to FBX
cd /workspace/Puppeteer
/workspace/blender-4.2.0-linux-x64/blender --background --python-expr "
import sys; sys.argv = ['', '--mesh', '/workspace/Puppeteer/examples/character.obj', '--rig', '/workspace/Puppeteer/results/final_rigging/character.txt', '--output', '/workspace/Puppeteer/character_rigged.fbx']
exec(open('/workspace/Puppeteer/export.py').read())
"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'trimesh'"

**Solution**: Make sure conda environment is activated:

```bash
conda activate puppeteer
which python  # Should show /opt/conda/envs/puppeteer/bin/python
```

### Issue: "Git submodules not found"

**Symptoms**: Errors about missing `third_partys/Michelangelo`

**Solution**: Initialize submodules:

```bash
cd /workspace/Puppeteer
git submodule update --init --recursive
```

Or clone fresh with `--recursive` flag.

### Issue: "CUDA out of memory"

**Symptoms**: `RuntimeError: CUDA out of memory` during animation

**Solution**: Reduce tracked vertices in `animation/utils/save_utils.py`:

```python
# Line 241-242
MAX_VISIBLE_POINTS = 5000  # Reduce from 15000
MAX_SAMPLE_POINTS = 2000   # Reduce from 4000
```

Then clean cache and re-run:

```bash
rm -rf examples/character/track_2d_verts
./demo_animation.sh
```

### Issue: "RuntimeError: The size of tensor a (960) must match the size of tensor b (720)"

**Symptoms**: Dimension mismatch during optimization

**Cause**: Video dimensions don't match `--img_size` parameter

**Solution**:

1. Check your video dimensions:
   ```bash
   ffmpeg -i examples/character/input.mp4 2>&1 | grep "Stream.*Video"
   ```
2. Update `demo_animation.sh` line 57 to match:
   ```bash
   --img_size 720  # Must match video width/height
   ```
3. Ensure video is square (720x720)

### Issue: Cached data from previous run causing errors

**Solution**: Clean all cached data:

```bash
cd /workspace/Puppeteer
rm -rf examples/character/imgs
rm -rf examples/character/depth
rm -rf examples/character/flow*
rm -rf examples/character/track_2d_*
rm -rf results/animation/character
```

### Issue: SCP/file download not working

**Cause**: RunPod SSH doesn't support SCP protocol

**Solutions**:

1. Use RunPod's web file browser (easiest)
2. Use HTTP server method (see [Exporting Results](#exporting-results))
3. Use tar + cat over SSH (see examples above)

### Issue: Video is not square (portrait/landscape)

**Symptoms**: Animation fails with dimension errors

**Solution**: Convert video to square before running:

```bash
ffmpeg -i input.mp4 \
  -vf "scale=720:720:force_original_aspect_ratio=decrease,pad=720:720:(ow-iw)/2:(oh-ih)/2" \
  output_720x720.mp4
```

---

## Running Animation with New Video

### Quick Clean & Re-run (video already uploaded)

```bash
# Clean cache and re-run animation with new video already in place
cd /workspace/Puppeteer && rm -rf examples/character/{imgs,depth,flow*,track_2d_*} results/animation/character && conda activate puppeteer && PYOPENGL_PLATFORM=egl ./demo_animation.sh
```

### Full Process (prepare video + clean + run)

```bash
# Prepare new video (make it 720x720), replace old video, clean cache, and re-run
ffmpeg -i new_video.mp4 -vf "scale=720:720:force_original_aspect_ratio=decrease,pad=720:720:(ow-iw)/2:(oh-ih)/2" new_720x720.mp4 && \
cp new_720x720.mp4 /workspace/Puppeteer/examples/character/input.mp4 && \
cd /workspace/Puppeteer && \
rm -rf examples/character/{imgs,depth,flow*,track_2d_*} results/animation/character && \
conda activate puppeteer && \
PYOPENGL_PLATFORM=egl ./demo_animation.sh
```

### What Gets Cleared

These cached directories MUST be deleted when using a new video:

- `examples/character/imgs` - Extracted frames from old video
- `examples/character/depth` - Depth estimation (wrong dimensions from old video)
- `examples/character/flow*` - Optical flow data
- `examples/character/track_2d_*` - Tracking data (joints and vertices)
- `results/animation/character` - Previous animation outputs

---

## Key File Locations

### Input Files

- `examples/{model}.obj` - Your 3D model
- `examples/{model}/input.mp4` - Animation reference video

### Output Files

- `results/final_rigging/{model}.txt` - Final rig (skeleton + weights)
- `results/skel_results/{model}_skel.obj` - Skeleton visualization
- `results/animation/{model}/{save_name}/concat_output.mp4` - Main result

### Cache Files (safe to delete)

- `examples/{model}/imgs/` - Extracted video frames
- `examples/{model}/flow*` - Optical flow data
- `examples/{model}/depth/` - Depth estimation
- `examples/{model}/track_2d_*` - Tracking data

---

## Tips & Best Practices

1. **Video Quality**: Use clear, well-lit videos showing the full character
2. **Video Length**: 2-5 seconds works well (10-50 frames at 10fps)
3. **Model Quality**: Clean, watertight meshes work best
4. **First Run**: Start with fewer iterations (`--iter 30`) to test, then increase
5. **GPU Memory**: Monitor with `nvidia-smi` - reduce tracking points if OOM
6. **Backup Results**: Copy important results to `/workspace/` for persistence
7. **Git Commits**: Commit your changes regularly if customizing the pipeline

---

## Common Workflows

### Workflow 1: Single Character, Multiple Animations

1. Run rigging once: `./demo_rigging.sh`
2. For each new video:
   - Replace `input.mp4`
   - Clean cache
   - Run `./demo_animation.sh`

### Workflow 2: Multiple Characters

1. Place all `.obj` files in `examples/`
2. Run rigging: processes all models
3. For each character:
   - Create `examples/{character}/` folder
   - Add `input.mp4`
   - Edit `demo_animation.sh` to set `--seq_name`
   - Run animation

### Workflow 3: Iterative Refinement

1. Run with `--iter 30` (quick test)
2. Check results
3. Adjust parameters (smoothing weights, etc.)
4. Run with `--iter 50` (final quality)

---

## Parameter Tuning

### Animation Optimization Parameters

In `demo_animation.sh` line 57:

- `--iter 50` - Number of iterations (30-100)
  - More = better quality but slower
- `--img_size 720` - Must match video dimensions
- `--smooth_weight 1` - Temporal smoothness (0.5-2.0)
  - Higher = smoother but less responsive
- `--main_renderer front_left` - Primary camera view
- `--additional_renderer "right,front_right"` - Extra views

### Memory Management

In `animation/utils/save_utils.py` line 241-242:

```python
MAX_VISIBLE_POINTS = 5000   # Threshold for sampling
MAX_SAMPLE_POINTS = 2000    # Number of tracked vertices
```

Reduce if getting OOM errors. 1000-2000 points is usually sufficient.

---

## Getting Help

If you encounter issues:

1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Check the logs in terminal output
3. Verify file locations and permissions
4. Ensure conda environment is activated
5. Check GPU memory: `nvidia-smi`

**Common Commands**:

```bash
# Check environment
conda env list
which python

# Check GPU
nvidia-smi

# Check file sizes
du -sh results/animation/character/character_demo/

# View logs
tail -f optimization.log
```
