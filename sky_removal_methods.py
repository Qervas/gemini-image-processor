"""
Gemini-powered sky removal methods for 2025
Focused on Google Gemini 2.5 Flash for sky segmentation
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from google_sky_removal import GoogleSkyRemoval
except ImportError:
    GoogleSkyRemoval = None

class SkyRemovalMethods:
    def __init__(self, google_api_key: str = None):
        self.google_api_key = google_api_key

        # Initialize Google Gemini sky removal
        if GoogleSkyRemoval:
            self.google_remover = GoogleSkyRemoval(google_api_key)
        else:
            self.google_remover = None

        print(" Gemini Sky Removal initialized")
        if self.google_remover:
            print(" Gemini sky removal available")

    def gemini_sky_removal(self, image_path: str, output_path: str = None, prompt_type: str = 'default', **prompt_kwargs) -> np.ndarray:
        """
        Gemini 2.5 Flash based sky removal - the best method

        Args:
            image_path: Path to input image
            output_path: Optional path to save result
            prompt_type: Type of prompt to use ('default', 'conservative', etc.)
            **prompt_kwargs: Additional arguments for prompt template processing
        """
        if self.google_remover:
            # Handle custom prompt type
            if prompt_type == 'custom' and 'custom_prompt' in prompt_kwargs:
                # For custom prompts, we'll need to modify the google_remover temporarily
                # or pass the custom prompt through kwargs
                return self.google_remover.gemini_sky_removal(
                    image_path, output_path, 'default',
                    custom_prompt=prompt_kwargs['custom_prompt']
                )
            else:
                return self.google_remover.gemini_sky_removal(image_path, output_path, prompt_type, **prompt_kwargs)
        else:
            raise ValueError("Gemini not available. Install google-generativeai and set GOOGLE_API_KEY")

    def find_images_recursive(self, directory: str) -> list:
        """
        Recursively find all image files in directory and subdirectories
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))

        return sorted(image_files)

    def process_images_recursive(self, images_dir: str = "images", output_base_dir: str = "results", prompt_type: str = 'default', **prompt_kwargs):
        """
        Process all images in directory recursively with Gemini

        Args:
            images_dir: Input directory containing images
            output_base_dir: Output directory for results
            prompt_type: Type of prompt to use ('default', 'conservative', etc.)
            **prompt_kwargs: Additional arguments for prompt template processing
        """
        if not os.path.exists(images_dir):
            print(f"Images directory '{images_dir}' not found!")
            return {}

        image_files = self.find_images_recursive(images_dir)
        if not image_files:
            print(f"No image files found in '{images_dir}'")
            return {}

        print(f"Found {len(image_files)} images to process with Gemini")

        all_results = {}
        for i, image_path in enumerate(image_files, 1):
            rel_path = os.path.relpath(image_path, images_dir)
            print(f"\n[{i}/{len(image_files)}] Processing: {rel_path}")

            # Create output directory preserving folder structure
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            subdir = os.path.dirname(rel_path)
            if subdir:
                output_dir = os.path.join(output_base_dir, subdir, image_name)
            else:
                output_dir = os.path.join(output_base_dir, image_name)

            try:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "gemini_result.jpg")
                result = self.gemini_sky_removal(image_path, output_path, prompt_type, **prompt_kwargs)
                all_results[image_path] = {"Gemini_2_5": result}
                print(f" Gemini processing completed for {rel_path}")
            except Exception as e:
                print(f" Failed to process {rel_path}: {e}")
                all_results[image_path] = {}

        print(f"\nProcessing complete! Results saved in '{output_base_dir}'")
        return all_results

    def compare_methods(self, image_path: str, output_dir: str = "results", prompt_type: str = 'default', **prompt_kwargs):
        """
        Compare sky removal methods - now only Gemini

        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            prompt_type: Type of prompt to use ('default', 'conservative', etc.)
            **prompt_kwargs: Additional arguments for prompt template processing
        """
        os.makedirs(output_dir, exist_ok=True)

        methods = [("Gemini_2_5", lambda img, out: self.gemini_sky_removal(img, out, prompt_type, **prompt_kwargs))]

        results = {}
        for name, method in methods:
            print(f"Testing {name}...")
            output_path = os.path.join(output_dir, f"{name}_result.jpg")
            try:
                result = method(image_path, output_path)
                results[name] = result
                print(f" {name} completed")
            except Exception as e:
                print(f" {name} failed: {e}")

        return results

if __name__ == "__main__":
    # Example usage
    sky_remover = SkyRemovalMethods()

    print("Gemini Sky Removal initialized. Use gemini_sky_removal() or compare_methods() on your images.")
