"""
Google Gemini-powered sky removal using Gemini 2.5 Flash
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")

# Import prompt processor
try:
    from prompt_processor import PromptProcessor
except ImportError:
    print("⚠️  prompt_processor.py not found. Using default prompts.")
    PromptProcessor = None

class GoogleSkyRemoval:
    def __init__(self, google_api_key: str = None):
        # Load API key from .env file if not provided
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.max_image_size = int(os.getenv('MAX_IMAGE_SIZE', '2048'))

        # Initialize prompt processor
        self.prompt_processor = PromptProcessor() if PromptProcessor else None

        print("🚀 Google Gemini Sky Removal initialized")
        if self.google_api_key:
            print("✅ Gemini API key loaded from environment")
        else:
            print("⚠️  No API key found. Set GOOGLE_API_KEY in .env file")

        if self.prompt_processor:
            print("📝 Prompt processor loaded")
        else:
            print("⚠️  Using default prompts (no prompt_processor.py found)")

    def _resize_image_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize image if it exceeds max size"""
        h, w = image.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h))
        return image

    def gemini_sky_removal(self, image_path: str, output_path: str = None, prompt_type: str = 'default', **prompt_kwargs) -> np.ndarray:
        """
        Gemini 2.5 Flash conversational image segmentation for sky removal

        Args:
            image_path: Path to input image
            output_path: Optional path to save result
            prompt_type: Type of prompt to use ('default', 'conservative', etc.)
            **prompt_kwargs: Additional arguments for prompt template processing
        """
        try:
            import google.generativeai as genai
            from PIL import Image

            if not self.google_api_key:
                raise ValueError("Google API key required for Gemini. Set GOOGLE_API_KEY in .env file")

            # Configure Gemini
            genai.configure(api_key=self.google_api_key)
            model = genai.GenerativeModel('models/gemini-2.5-flash-image-preview')

            # Load image
            image = Image.open(image_path)

            # Get processed prompt
            if 'custom_prompt' in prompt_kwargs:
                # Use custom prompt directly
                prompt = prompt_kwargs['custom_prompt']
                print(f"📝 Using custom prompt")
            elif self.prompt_processor:
                prompt = self.prompt_processor.get_prompt(prompt_type, **{k: v for k, v in prompt_kwargs.items() if k != 'custom_prompt'})
            else:
                # Fallback to default prompt if processor not available
                prompt = """
                Remove the sky from this image and replace it with a transparent background.
                Make sure to preserve all buildings, trees, people, and other ground-level objects.
                Use the surrounding colors and lighting to create a natural-looking result.
                """

            print(f"📝 Using prompt type: {prompt_type}")
            print(f"📝 Prompt: {prompt[:100]}...")

            # Generate response
            response = model.generate_content([prompt, image])
            print(f"📄 Gemini response received")

            # Handle different response types
            generated_image = None
            response_text = ""

            if hasattr(response, 'parts') and response.parts:
                print(f"� Response has {len(response.parts)} parts")

                for i, part in enumerate(response.parts):
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
                        print(f"📝 Part {i} text: {part.text[:100]}...")

                    if hasattr(part, 'inline_data') and part.inline_data:
                        # This is an image response
                        try:
                            from PIL import Image as PILImage
                            import io

                            image_data = part.inline_data.data
                            generated_image = PILImage.open(io.BytesIO(image_data))
                            print(f"🖼️  Part {i} contains generated image: {generated_image.size}")
                        except Exception as img_error:
                            print(f"⚠️  Could not process image data in part {i}: {img_error}")

            # Try legacy text extraction
            try:
                if not response_text:
                    response_text = response.text
                    print(f"📝 Legacy text response: {response_text[:200]}...")
            except Exception as text_error:
                print(f"⚠️  Could not extract text: {text_error}")

            # Process the result
            if generated_image:
                print("✅ Gemini generated an edited image!")

                # Convert PIL image to numpy array
                if generated_image.mode != 'RGB':
                    generated_image = generated_image.convert('RGB')

                result_array = np.array(generated_image)

                if output_path:
                    # Save the generated image
                    generated_image.save(output_path)
                    print(f"✅ Saved Gemini-generated image to: {output_path}")

                return result_array

            else:
                print("⚠️  Gemini provided analysis but no edited image")
                print(f"💡 Response: {response_text[:300]}...")

                # Fall back to original image
                original_image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                if output_path:
                    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                    print(f"✅ Saved original image to: {output_path}")

                return image_rgb

        except ImportError:
            print("Google Generative AI not available. Install with: pip install google-generativeai")
            raise
        except Exception as e:
            print(f"Gemini segmentation failed: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    google_remover = GoogleSkyRemoval()

    print("Google Gemini Sky Removal available:")
    print("gemini_sky_removal() - Gemini 2.5 Flash segmentation")
    print("\nSet GOOGLE_API_KEY in .env file for Gemini features.")