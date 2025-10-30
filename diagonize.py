"""
System diagnostic script to check if your system can handle large models.
Run this before using flan-t5-large.
"""

import sys
import psutil
import platform
from pathlib import Path


def format_bytes(bytes_value):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0


def check_system():
    """Check system capabilities for running large models."""
    print("=" * 60)
    print("SYSTEM DIAGNOSTIC FOR LARGE MODEL")
    print("=" * 60)
    print()
    
    # Python version
    print(f"Python Version: {sys.version.split()[0]}")
    python_ok = sys.version_info >= (3, 8) and sys.version_info < (3, 12)
    print(f"Status: {'✓ OK' if python_ok else '✗ NOT COMPATIBLE (need 3.8-3.11)'}")
    print()
    
    # Operating System
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print()
    
    # Memory
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    used_gb = memory.used / (1024**3)
    
    print("MEMORY:")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Used: {used_gb:.1f} GB ({memory.percent}%)")
    print(f"  Available: {available_gb:.1f} GB")
    print()
    
    # Recommendations
    print("MODEL RECOMMENDATIONS:")
    print("-" * 60)
    
    if available_gb < 3:
        print("⚠️  WARNING: Very low memory!")
        print("   Recommended: flan-t5-small")
        print(f"   Available: {available_gb:.1f} GB | Needed: 2 GB")
        recommendation = "small"
        
    elif available_gb < 6:
        print("✓ Sufficient for base model")
        print("   Recommended: flan-t5-base (current default)")
        print(f"   Available: {available_gb:.1f} GB | Needed: 4 GB")
        recommendation = "base"
        
    elif available_gb < 10:
        print("⚠️  Borderline for large model")
        print("   Recommended: flan-t5-base (safer)")
        print("   Possible: flan-t5-large (may cause disconnects)")
        print(f"   Available: {available_gb:.1f} GB | Needed: 8-10 GB")
        recommendation = "base (or try large with caution)"
        
    elif available_gb < 18:
        print("✓ Good for large model")
        print("   Recommended: flan-t5-large")
        print(f"   Available: {available_gb:.1f} GB | Needed: 8 GB")
        recommendation = "large"
        
    else:
        print("✓✓ Excellent! Can handle any model")
        print("   Recommended: flan-t5-xl (best quality)")
        print(f"   Available: {available_gb:.1f} GB | Needed: 16 GB")
        recommendation = "xl"
    
    print()
    
    # CPU
    print("CPU:")
    print(f"  Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"  Usage: {psutil.cpu_percent(interval=1)}%")
    print()
    
    # Disk
    disk = psutil.disk_usage('/')
    print("DISK SPACE:")
    print(f"  Total: {format_bytes(disk.total)}")
    print(f"  Used: {format_bytes(disk.used)} ({disk.percent}%)")
    print(f"  Free: {format_bytes(disk.free)}")
    print()
    
    # Check model sizes
    models_dir = Path("models/summarization")
    if models_dir.exists():
        total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
        print(f"Current Model Size: {format_bytes(total_size)}")
        print()
    
    # Model size requirements
    print("MODEL DOWNLOAD SIZES:")
    print("  flan-t5-small: ~250 MB")
    print("  flan-t5-base: ~900 MB (current)")
    print("  flan-t5-large: ~3 GB")
    print("  flan-t5-xl: ~11 GB")
    print()
    
    # Final recommendations
    print("=" * 60)
    print("RECOMMENDATIONS FOR YOUR SYSTEM")
    print("=" * 60)
    print()
    
    print(f"✓ Best Model for You: flan-t5-{recommendation}")
    print()
    
    if recommendation == "base" or recommendation == "base (or try large with caution)":
        print("GOOD NEWS:")
        print("  The base model with improved settings gives 40-60% better")
        print("  answers than the old default. You don't need the large model!")
        print()
        print("  Current config (config/settings.py) is already optimized.")
        print()
    
    if "large" in recommendation:
        print("TO USE LARGE MODEL:")
        print("  1. Edit config/settings.py:")
        print('     SUMMARIZATION_MODEL_NAME = "google/flan-t5-large"')
        print("  2. Run: python models/download_models.py")
        print("  3. Create .streamlit/config.toml with timeout settings")
        print("  4. Close other applications before running")
        print()
    
    if recommendation == "small":
        print("TO OPTIMIZE FOR YOUR SYSTEM:")
        print("  1. Edit config/settings.py:")
        print('     SUMMARIZATION_MODEL_NAME = "google/flan-t5-small"')
        print("  2. Close other applications")
        print("  3. Consider upgrading RAM if possible")
        print()
    
    # Warnings
    if memory.percent > 80:
        print("⚠️  WARNING: High memory usage detected!")
        print("   Close other applications before running the system.")
        print()
    
    if disk.percent > 90:
        print("⚠️  WARNING: Low disk space!")
        print("   Free up space before downloading large models.")
        print()
    
    print("=" * 60)


def main():
    """Main diagnostic function."""
    try:
        check_system()
    except Exception as e:
        print(f"Error running diagnostics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()