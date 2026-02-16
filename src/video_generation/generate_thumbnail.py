#!/usr/bin/env python3
"""
Thumbnail Generation - Uses FastSD CPU to generate video thumbnails locally
"""
import json
import argparse
import subprocess
import sys
from pathlib import Path

def generate_thumbnail(script_file, run_id):
    """Generate thumbnail using FastSD CPU locally"""
    
    print("📂 Loading script...")
    
    # Load script
    try:
        with open(script_file, 'r', encoding='utf-8') as f:
            script_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load script file: {e}")
        return None
    
    # Get thumbnail prompt
    prompt = script_data.get('metadata', {}).get('stability_ai_prompt', '')
    
    if not prompt:
        # Fallback to title
        title = script_data.get('metadata', {}).get('final_title', '')
        prompt = title if title else "YouTube thumbnail"
    
    # Enhanced prompt for viral YouTube thumbnail quality
    enhanced_prompt = f"""{prompt}, YouTube thumbnail, Indian male face close up, extreme shocked expression, wide open eyes, cinematic lighting, dramatic shadows, dark psychology theme, high contrast, ultra realistic, DSLR photography, sharp focus, vibrant colors, viral clickbait style, centered composition, subject filling frame, 8k resolution"""
    
    # High quality negative prompt
    negative_prompt = """blurry, low quality, distorted face, extra fingers, extra limbs, duplicate face, bad anatomy, watermark, logo, text, cropped, out of frame, dull colors, low contrast"""
    
    # Setup output path
    output_dir = Path.cwd() / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'thumbnail.jpg'
    
    print("🎨 Generating thumbnail using FastSD CPU...")
    print(f"   Prompt: {enhanced_prompt[:100]}...")
    
    try:
        # FastSD CPU path resolution - absolute path for reliability
        fastsdcpu_path = Path.cwd() / "fastsdcpu" / "src" / "app.py"
        
        if not fastsdcpu_path.exists():
            print(f"❌ ERROR: FastSD CPU not found at {fastsdcpu_path}")
            return None
        
        print(f"Using FastSD CPU at: {fastsdcpu_path}")
        
        cmd = [
            sys.executable,
            str(fastsdcpu_path),
            "--prompt", enhanced_prompt,
            "--negative_prompt", negative_prompt,
            "--width", "1024",
            "--height", "1024",
            "--steps", "6",  # Increased for better quality while maintaining speed
            "--output", str(output_file)
        ]
        
        print("🚀 Running FastSD CPU...")
        
        # Run FastSD CPU with timeout protection
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300  # 5 minute timeout for GitHub Actions
        )
        
        if result.returncode != 0:
            print("❌ FastSD CPU execution failed")
            print(f"   Return code: {result.returncode}")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            return None
            
        # Verify output file was created and has content
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"✅ Thumbnail successfully generated: {output_file}")
            return output_file
        else:
            print("❌ Thumbnail generation failed or file empty")
            if output_file.exists():
                print(f"   File exists but size is {output_file.stat().st_size} bytes")
            else:
                print("   Output file not found")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ FastSD CPU execution timed out after 300 seconds")
        return None
    except Exception as e:
        print(f"❌ Thumbnail generation failed with unexpected error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate YouTube thumbnail using FastSD CPU')
    parser.add_argument('--script-file', required=True, help='Path to script JSON file')
    parser.add_argument('--run-id', required=True, help='Run ID for tracking')
    args = parser.parse_args()
    
    generate_thumbnail(args.script_file, args.run_id)

if __name__ == '__main__':
    main()