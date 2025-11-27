# How to Use CN-CLIP (Chinese CLIP) with CLIP-as-service

## Overview

CN-CLIP is the Chinese version of CLIP, designed for Chinese language understanding and cross-modal search. CLIP-as-service supports CN-CLIP models natively.

## Installation

### Step 1: Install CLIP Server with CN-CLIP Support

```bash
pip install "clip-server[cn_clip]"
```

Or install separately:
```bash
pip install clip-server
pip install cn_clip
```

## Available CN-CLIP Models

The following CN-CLIP models are supported:

- `CN-CLIP/ViT-B-16` - Base Vision Transformer (16 patches)
- `CN-CLIP/ViT-L-14` - Large Vision Transformer (14 patches)
- `CN-CLIP/ViT-L-14-336` - Large Vision Transformer (336px resolution)
- `CN-CLIP/ViT-H-14` - Huge Vision Transformer (14 patches)
- `CN-CLIP/RN50` - ResNet-50 backbone

## Usage Methods

### Method 1: Using YAML Configuration File (Recommended)

1. Create a configuration file `cnclip-flow.yml`:

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
        name: CN-CLIP/ViT-B-16  # Choose your model
        device: cuda             # or 'cpu'
        minibatch_size: 32       # Adjust based on your GPU memory
    timeout_ready: 3000000
    replicas: 1
```

2. Start the server:

```bash
python -m clip_server cnclip-flow.yml
```

### Method 2: Modify Default Configuration

You can also modify the default `torch-flow.yml` to include the `name` parameter:

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
        name: CN-CLIP/ViT-B-16  # Add this line
    timeout_ready: 3000000
    replicas: 1
```

### Method 3: Programmatic Setup

Create a Python script to start the server:

```python
from jina import Flow
from clip_server.executors.clip_torch import CLIPEncoder

f = Flow(port=51000).add(
    uses=CLIPEncoder,
    uses_with={'name': 'CN-CLIP/ViT-B-16', 'device': 'cuda'}
)

with f:
    f.block()
```

## Important Notes

### Text Encoding

CN-CLIP uses a different tokenizer and context length:

- **Context Length**: 52 (instead of 77 for English CLIP)
- **Tokenizer**: Uses `cn_clip.clip` tokenizer
- **Text Handling**: Automatically handles Chinese text encoding

### Model Configuration Parameters

You can configure the executor with these parameters:

- `name`: Model name (required) - e.g., `CN-CLIP/ViT-B-16`
- `device`: Device to use - `'cuda'` or `'cpu'` (default: auto-detect)
- `dtype`: Data type - `'fp16'`, `'fp32'` (default: fp16 for CUDA, fp32 for CPU)
- `minibatch_size`: Batch size for processing (default: 32)
- `num_worker_preprocess`: Number of preprocessing workers (default: 4)

## Client Usage

Once the server is running, use the client normally:

```python
from clip_client import Client
from docarray import Document

# Connect to server
c = Client('grpc://0.0.0.0:51000')

# Encode Chinese text
result = c.encode(['这是一只猫', '这是一只狗', '这是一只鸟'])
print(result.shape)  # [3, embedding_dim]

# Encode images
result = c.encode(['image1.jpg', 'image2.png'])
print(result.shape)

# Cross-modal search
# (same API as regular CLIP)
```

## Troubleshooting

1. **Import Error**: Make sure `cn_clip` is installed:
   ```bash
   pip install cn_clip
   ```

2. **Model Loading**: CN-CLIP models are loaded from the `cn_clip` package, so they will be downloaded automatically on first use.

3. **Memory Issues**: If you encounter OOM errors, reduce `minibatch_size` in the configuration.

4. **Performance**: CN-CLIP models work best with GPU. Ensure CUDA is available if using GPU mode.

## Code Reference

- Model implementation: `server/clip_server/model/cnclip_model.py`
- Tokenizer: `server/clip_server/model/tokenization.py` (lines 16-19, 40-41, 68-78)
- Model registry: `server/clip_server/model/pretrained_models.py` (lines 104-110)

