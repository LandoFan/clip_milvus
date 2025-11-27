#!/usr/bin/env python
"""Test script to check available CN-CLIP model names"""

try:
    import cn_clip.clip as cnclip
    print("✓ cn_clip package is installed")
    
    # Try to find available models
    if hasattr(cnclip, 'available_models'):
        models = cnclip.available_models()
        print(f"\nAvailable models from cn_clip: {models}")
    
    # Try common model name formats
    test_names = [
        'ViT-B-16',
        'ViT-L-14', 
        'ViT-L-14-336',
        'ViT-H-14',
        'RN50',
        'cn-clip/vit-b-16',
        'CN-CLIP/ViT-B-16',
    ]
    
    print("\nTesting model name formats that cn_clip might accept:")
    print("-" * 60)
    
    # Check what load_from_name expects
    import inspect
    sig = inspect.signature(cnclip.load_from_name)
    print(f"\nload_from_name signature: {sig}")
    
    # Try to see the docstring
    if cnclip.load_from_name.__doc__:
        print(f"\nDocstring:\n{cnclip.load_from_name.__doc__[:500]}")
        
except ImportError as e:
    print(f"✗ Error importing cn_clip: {e}")
    print("\nPlease install cn_clip first:")
    print("  pip install cn_clip")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

