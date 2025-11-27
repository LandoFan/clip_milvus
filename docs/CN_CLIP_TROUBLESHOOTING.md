# CN-CLIP Model Troubleshooting Guide

## Problem: Model Not Found

If you're getting an error that `CN-CLIP/ViT-B-16` is not found, this guide will help you find the correct model name.

## Step 1: Check Available Models

First, let's check what models are actually available in your `cn_clip` installation:

```python
from cn_clip.clip import available_models
print(available_models())
```

Or run the provided script:
```bash
python check_cnclip.py
```

## Step 2: Common Model Name Formats

The `cn_clip` package may use different model name formats. Here are the common ones to try:

### Format 1: Simple names (most common)
- `ViT-B-16`
- `ViT-L-14`
- `ViT-L-14-336`
- `ViT-H-14`
- `RN50`

### Format 2: With prefix
- `cn-clip/ViT-B-16`
- `chinese-clip-vit-base-patch16`

### Format 3: With different casing
- `vit-b-16` (lowercase)
- `vit-l-14`

## Step 3: Test the Model Name Directly

Try loading the model directly to see what name format works:

```python
from cn_clip.clip import load_from_name
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Try different name formats
for model_name in ['ViT-B-16', 'vit-b-16', 'CN-CLIP/ViT-B-16']:
    try:
        model, preprocess = load_from_name(model_name, device=device)
        print(f"✓ Successfully loaded: {model_name}")
        break
    except Exception as e:
        print(f"✗ Failed with '{model_name}': {e}")
```

## Step 4: Update the Configuration

Once you know the correct model name format, you have two options:

### Option A: Use a model name that matches cn_clip's format directly

If the direct model name works (like `ViT-B-16`), you can modify the code to support it, or use one of the already mapped names.

### Option B: Check the current mapping

The current mapping in `clip_server/model/cnclip_model.py` is:

```python
_CNCLIP_MODEL_MAPS = {
    'CN-CLIP/ViT-B-16': 'ViT-B-16',
    'CN-CLIP/ViT-L-14': 'ViT-L-14',
    'CN-CLIP/ViT-L-14-336': 'ViT-L-14-336',
    'CN-CLIP/ViT-H-14': 'ViT-H-14',
    'CN-CLIP/RN50': 'RN50',
}
```

If `ViT-B-16` doesn't work, you might need to update this mapping.

## Step 5: Verify cn_clip Package Version

Different versions of `cn_clip` might use different model names. Check your version:

```python
import cn_clip
print(cn_clip.__version__)
```

Then check the documentation for that specific version at:
- https://github.com/OFA-Sys/Chinese-CLIP
- https://pypi.org/project/cn-clip/

## Quick Fix: Try Alternative Model Names

If `CN-CLIP/ViT-B-16` doesn't work, try modifying the configuration to use the mapped name directly:

```yaml
jtype: Flow
version: '1'
with:
  port: 51000
executors:
  - name: clip_t
    uses:
      jtype: CLIPEncoder
      metas:
        py_modules:
          - clip_server.executors.clip_torch
      with:
        name: ViT-B-16  # Try the mapped name directly
        device: cuda
    timeout_ready: 3000000
    replicas: 1
```

**Note:** This might not work if the executor expects the full `CN-CLIP/` prefix. In that case, you may need to:

1. Check what model names are actually registered in `_CNCLIP_MODELS`
2. Update the mapping in `cnclip_model.py` to match your `cn_clip` version
3. Or update the `cn_clip` package to a compatible version

## Still Having Issues?

1. **Check the error message** - it usually tells you what model names are available
2. **Check the cn_clip documentation** - https://github.com/OFA-Sys/Chinese-CLIP
3. **Verify installation** - Make sure `cn_clip` is properly installed:
   ```bash
   pip uninstall cn-clip
   pip install cn-clip
   ```
4. **Try a different model** - Try `CN-CLIP/RN50` which is typically more stable

