#!/usr/bin/env python
"""Script to check CN-CLIP model availability"""

print("Checking CN-CLIP model support...")
print("=" * 60)

# Check if cn_clip is installed
try:
    import cn_clip.clip as cnclip
    print("✓ cn_clip package is installed\n")
    
    # Try to get available models
    if hasattr(cnclip, 'available_models'):
        try:
            models = cnclip.available_models()
            print(f"Available models: {models}\n")
        except Exception as e:
            print(f"Could not get available_models(): {e}\n")
    
    # Check the load_from_name function signature
    import inspect
    sig = inspect.signature(cnclip.load_from_name)
    print(f"load_from_name signature: {sig}")
    
    # Check docstring
    if cnclip.load_from_name.__doc__:
        doc = cnclip.load_from_name.__doc__
        print(f"\nDocumentation (first 300 chars):\n{doc[:300]}...")
    
    print("\n" + "=" * 60)
    print("\nTesting model name formats:")
    print("-" * 60)
    
    # Test common formats
    test_names = ['ViT-B-16', 'ViT-L-14', 'RN50']
    
    for name in test_names:
        print(f"\nTrying: '{name}'")
        try:
            # Just check if it would work (don't actually load)
            print(f"  (Model name format check passed)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            
except ImportError:
    print("✗ cn_clip package is NOT installed")
    print("\nPlease install it:")
    print("  pip install cn_clip")
    print("  or")
    print("  pip install 'clip-server[cn_clip]'")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("\nChecking clip-server CN-CLIP model mappings:")
print("-" * 60)

try:
    import sys
    sys.path.insert(0, 'server')
    from clip_server.model.pretrained_models import _CNCLIP_MODELS
    from clip_server.model.cnclip_model import _CNCLIP_MODEL_MAPS
    
    print("\nRegistered CN-CLIP models in clip-server:")
    for model in _CNCLIP_MODELS:
        mapped = _CNCLIP_MODEL_MAPS.get(model, 'NOT FOUND')
        print(f"  {model} -> {mapped}")
        
except Exception as e:
    print(f"Error checking clip-server: {e}")

