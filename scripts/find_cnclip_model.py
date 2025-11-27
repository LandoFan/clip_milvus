#!/usr/bin/env python
"""Script to find the correct CN-CLIP model names for your environment"""

import sys

print("=" * 70)
print("CN-CLIP Model Name Finder")
print("=" * 70)
print()

# Step 1: Check if cn_clip is installed
try:
    import cn_clip.clip as cnclip
    print("✓ cn_clip package is installed")
    
    # Try to get version info
    try:
        import cn_clip
        version = getattr(cn_clip, '__version__', 'unknown')
        print(f"  Version: {version}")
    except:
        pass
    
    print()
except ImportError:
    print("✗ cn_clip package is NOT installed")
    print("\nPlease install it first:")
    print("  pip install cn_clip")
    print("  or")
    print("  pip install 'clip-server[cn_clip]'")
    sys.exit(1)

# Step 2: Try to get available models
print("Step 1: Checking available models...")
print("-" * 70)

available = None
if hasattr(cnclip, 'available_models'):
    try:
        available = cnclip.available_models()
        print(f"✓ Found available_models() function")
        print(f"  Available models: {available}")
        print()
    except Exception as e:
        print(f"✗ Could not call available_models(): {e}")
        print()

# Step 3: Test different model name formats
print("Step 2: Testing model name formats...")
print("-" * 70)

# Common model names to try
test_names = [
    # Standard formats
    'ViT-B-16',
    'ViT-L-14',
    'ViT-L-14-336',
    'ViT-H-14',
    'RN50',
    # Alternative formats
    'vit-b-16',
    'ViT-B/16',
    'vit-b/16',
    'chinese-clip-vit-base-patch16',
]

working_names = []

import torch
device = "cpu"  # Use CPU for testing to avoid GPU requirements

for model_name in test_names:
    try:
        print(f"  Testing '{model_name}'...", end=" ")
        model, preprocess = cnclip.load_from_name(model_name, device=device)
        print("✓ SUCCESS")
        working_names.append(model_name)
        del model  # Free memory
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() or "unknown" in error_msg.lower():
            print("✗ Not found")
        else:
            print(f"✗ Error: {error_msg[:50]}")

print()

# Step 4: Show results
print("=" * 70)
print("RESULTS")
print("=" * 70)

if working_names:
    print("\n✓ Working model names:")
    for name in working_names:
        print(f"  - {name}")
    
    print("\n💡 Recommendation:")
    recommended = working_names[0]
    print(f"  Use: '{recommended}'")
    print("\n  Update your configuration file with:")
    print(f"    name: {recommended}")
    
    # Check if it matches the expected format
    if recommended.startswith('ViT') or recommended.startswith('RN'):
        print("\n  Or use in clip-server format:")
        clip_format = f"CN-CLIP/{recommended}"
        print(f"    name: {clip_format}")
        print(f"    (This will be mapped to: {recommended})")
else:
    print("\n✗ No working model names found!")
    print("\nPossible issues:")
    print("  1. Models need to be downloaded first")
    print("  2. Your cn_clip version uses different naming")
    print("  3. Network connectivity issues")
    
    if available:
        print(f"\nTry these from available_models(): {available}")

print()
print("=" * 70)

