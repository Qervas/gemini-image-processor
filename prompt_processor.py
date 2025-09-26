"""
Prompt Processing Module for Gemini Sky Removal

Handles prompt loading, customization, and processing for Gemini API calls.
Supports template substitution and prompt optimization.
"""

import os
import re
import json
from typing import Dict, Optional, List
from pathlib import Path

class PromptProcessor:
    """Handles prompt management and processing for Gemini sky removal"""

    _instance = None

    def __new__(cls, prompts_file: str = "prompts.json"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, prompts_file: str = "prompts.json"):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.prompts_file = Path(prompts_file)
        self.prompts_data = {}
        self.settings = {}

        # Load prompts from JSON file
        self._load_prompts_from_json()

        print("📝 Prompt Processor initialized")
        print(f"   File: {self.prompts_file}")
        print(f"   Language: {self.settings.get('language', 'english')}")
        print(f"   Optimization: {'enabled' if self.settings.get('enable_optimization', True) else 'disabled'}")
        print(f"   Max length: {self.settings.get('max_prompt_length', 500)} characters")

    def _load_prompts_from_json(self):
        """Load prompts from JSON file"""
        try:
            if not self.prompts_file.exists():
                print(f"⚠️  Prompts file not found: {self.prompts_file}")
                self._create_default_prompts()
                return

            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.prompts_data = data.get('prompts', {})
            self.settings = data.get('settings', {})

            print(f"✅ Loaded {len(self.prompts_data)} prompts from {self.prompts_file}")

        except Exception as e:
            print(f"⚠️  Error loading prompts from JSON: {e}")
            self._create_default_prompts()

    def _create_default_prompts(self):
        """Create default prompts if JSON file doesn't exist"""
        print("📝 Creating default prompts...")

        self.prompts_data = {
            "default": {
                "description": "Default sky removal optimized for RealityScan/photogrammetric methods",
                "prompt": "Remove the sky and clouds from this image and replace them with solid black background. This is for photogrammetric processing and 3D reconstruction. Preserve all buildings, trees, people, vehicles, and ground-level objects completely intact. Set sky areas to pure black (RGB 0,0,0) for clean masking in RealityScan workflows.",
                "use_case": "photogrammetry",
                "background_color": "black"
            },
            "conservative": {
                "description": "Conservative approach with minimal sky removal",
                "prompt": "Carefully remove only the sky from this image, keeping all foreground elements intact. Replace the sky with transparency while maintaining the original lighting and color balance.",
                "use_case": "safe_removal",
                "background_color": "transparent"
            }
        }

        self.settings = {
            "enable_optimization": True,
            "max_prompt_length": 500,
            "language": "english",
            "default_background_color": "black"
        }

    def get_prompt(self, prompt_type: str = 'default', **kwargs) -> str:
        """
        Get a processed prompt by type with optional template substitution

        Args:
            prompt_type: Type of prompt to use ('default', 'conservative', etc.)
            **kwargs: Template variables for substitution

        Returns:
            Processed prompt string
        """
        if prompt_type not in self.prompts_data:
            print(f"⚠️  Unknown prompt type '{prompt_type}', using default")
            prompt_type = 'default'

        prompt_data = self.prompts_data[prompt_type]
        prompt = prompt_data.get('prompt', '')

        # Process template substitutions
        if kwargs:
            prompt = self._process_template(prompt, kwargs)

        # Optimize prompt if enabled
        if self.settings.get('enable_optimization', True):
            prompt = self._optimize_prompt(prompt)

        return prompt

    def _process_template(self, template: str, variables: Dict) -> str:
        """
        Process template variables in prompt

        Args:
            template: Prompt template with {variable} placeholders
            variables: Dictionary of variable values

        Returns:
            Processed prompt with variables substituted
        """
        try:
            # Use string format for simple substitution
            processed = template.format(**variables)

            # Handle any remaining template variables with defaults
            processed = self._handle_missing_variables(processed)

            return processed

        except KeyError as e:
            print(f"⚠️  Missing template variable: {e}")
            return self._handle_missing_variables(template)
        except Exception as e:
            print(f"⚠️  Template processing error: {e}")
            return template

    def _handle_missing_variables(self, template: str) -> str:
        """
        Handle missing template variables by providing defaults
        """
        # Replace common missing variables with defaults
        defaults = {
            'image_description': 'an outdoor scene with sky and foreground objects',
            'scene_type': 'outdoor scene',
            'preservation_focus': 'buildings and natural elements',
            'removal_intensity': 'moderate'
        }

        for var, default in defaults.items():
            template = template.replace(f'{{{var}}}', default)

        return template

    def _optimize_prompt(self, prompt: str) -> str:
        """
        Optimize prompt for better Gemini performance

        Args:
            prompt: Original prompt

        Returns:
            Optimized prompt
        """
        max_length = self.settings.get('max_prompt_length', 500)
        if len(prompt) > max_length:
            # Truncate if too long
            prompt = prompt[:max_length - 3] + "..."
            print(f"⚠️  Prompt truncated to {max_length} characters")

        # Clean up whitespace
        prompt = re.sub(r'\s+', ' ', prompt.strip())

        # Ensure proper punctuation
        if not prompt.endswith(('.', '!', '?')):
            prompt += '.'

        return prompt

    def list_available_prompts(self) -> List[str]:
        """
        Get list of available prompt types

        Returns:
            List of prompt type names
        """
        return list(self.prompts_data.keys())

    def get_prompt_info(self, prompt_type: str = None) -> Dict:
        """
        Get information about prompts

        Args:
            prompt_type: Specific prompt type, or None for all

        Returns:
            Dictionary with prompt information
        """
        if prompt_type:
            if prompt_type in self.prompts_data:
                prompt_data = self.prompts_data[prompt_type]
                return {
                    'type': prompt_type,
                    'description': prompt_data.get('description', ''),
                    'use_case': prompt_data.get('use_case', ''),
                    'background_color': prompt_data.get('background_color', ''),
                    'length': len(prompt_data.get('prompt', ''))
                }
            else:
                return {'error': f'Prompt type "{prompt_type}" not found'}
        else:
            return {
                'available_prompts': self.list_available_prompts(),
                'settings': self.settings,
                'total_prompts': len(self.prompts_data)
            }

    def create_custom_prompt(self, base_type: str = 'default', modifications: Dict = None) -> str:
        """
        Create a custom prompt based on existing type with modifications

        Args:
            base_type: Base prompt type to modify
            modifications: Dictionary of modifications to apply

        Returns:
            Custom prompt string
        """
        if base_type not in self.prompts_data:
            base_type = 'default'

        prompt_data = self.prompts_data[base_type]
        prompt = prompt_data.get('prompt', '')

        if modifications:
            # Apply intensity modifications
            if 'intensity' in modifications:
                intensity = modifications['intensity'].lower()
                if intensity == 'conservative':
                    prompt = prompt.replace('Remove', 'Carefully remove')
                elif intensity == 'aggressive':
                    prompt = prompt.replace('Remove', 'Aggressively remove')

            # Apply background color modifications
            if 'background_color' in modifications:
                bg_color = modifications['background_color']
                if bg_color == 'black':
                    prompt = prompt.replace('transparent background', 'solid black background')
                    prompt = prompt.replace('transparency', 'pure black (RGB 0,0,0)')
                elif bg_color == 'white':
                    prompt = prompt.replace('transparent background', 'solid white background')

        return prompt

    def save_prompts_to_json(self):
        """Save current prompts back to JSON file"""
        try:
            data = {
                'prompts': self.prompts_data,
                'settings': self.settings,
                'metadata': {
                    'version': '1.0',
                    'model': 'gemini-2.5-flash',
                    'created': '2025-09-23',
                    'description': 'Prompt configurations for Gemini-powered sky removal'
                }
            }

            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"✅ Prompts saved to {self.prompts_file}")

        except Exception as e:
            print(f"❌ Error saving prompts: {e}")

# Global instance for easy access
prompt_processor = PromptProcessor()

if __name__ == "__main__":
    # Example usage
    processor = PromptProcessor()

    print("\n📝 Available Prompts:")
    for prompt_type in processor.list_available_prompts():
        info = processor.get_prompt_info(prompt_type)
        print(f"  {prompt_type}: {info['description']} ({info['length']} chars)")

    print("\n🔧 Settings:")
    settings = processor.get_prompt_info()['settings']
    for key, value in settings.items():
        print(f"  {key}: {value}")

    print("\n🎯 Prompt Examples:")
    print("Default:", processor.get_prompt('default')[:100] + "...")
    print("Conservative:", processor.get_prompt('conservative')[:100] + "...")

    print("\n🎨 Template Processing:")
    custom = processor.get_prompt('custom_template', image_description='a beautiful sunset over mountains')
    print("Custom:", custom)

    def get_prompt(self, prompt_type: str = 'default', **kwargs) -> str:
        """
        Get a processed prompt by type with optional template substitution

        Args:
            prompt_type: Type of prompt ('default', 'conservative', 'aggressive', etc.)
            **kwargs: Template variables for substitution

        Returns:
            Processed prompt string
        """
        if prompt_type not in self.default_prompts:
            print(f"⚠️  Unknown prompt type '{prompt_type}', using default")
            prompt_type = 'default'

        prompt = self.default_prompts[prompt_type]

        # Process template substitutions
        if kwargs:
            prompt = self._process_template(prompt, kwargs)

        # Optimize prompt if enabled
        if self.enable_optimization:
            prompt = self._optimize_prompt(prompt)

        return prompt

    def _process_template(self, template: str, variables: Dict) -> str:
        """
        Process template variables in prompt

        Args:
            template: Prompt template with {variable} placeholders
            variables: Dictionary of variable values

        Returns:
            Processed prompt with variables substituted
        """
        try:
            # Use string format for simple substitution
            processed = template.format(**variables)

            # Handle any remaining template variables with defaults
            processed = self._handle_missing_variables(processed)

            return processed

        except KeyError as e:
            print(f"⚠️  Missing template variable: {e}")
            return self._handle_missing_variables(template)
        except Exception as e:
            print(f"⚠️  Template processing error: {e}")
            return template

    def _handle_missing_variables(self, template: str) -> str:
        """
        Handle missing template variables by providing defaults
        """
        # Replace common missing variables with defaults
        defaults = {
            'image_description': 'an outdoor scene with sky and foreground objects',
            'scene_type': 'outdoor scene',
            'preservation_focus': 'buildings and natural elements',
            'removal_intensity': 'moderate'
        }

        for var, default in defaults.items():
            template = template.replace(f'{{{var}}}', default)

        return template

    def _optimize_prompt(self, prompt: str) -> str:
        """
        Optimize prompt for better Gemini performance

        Args:
            prompt: Original prompt

        Returns:
            Optimized prompt
        """
        if len(prompt) > self.max_prompt_length:
            # Truncate if too long
            prompt = prompt[:self.max_prompt_length - 3] + "..."
            print(f"⚠️  Prompt truncated to {self.max_prompt_length} characters")

        # Clean up whitespace
        prompt = re.sub(r'\s+', ' ', prompt.strip())

        # Ensure proper punctuation
        if not prompt.endswith(('.', '!', '?')):
            prompt += '.'

        return prompt

    def list_available_prompts(self) -> List[str]:
        """
        Get list of available prompt types

        Returns:
            List of prompt type names
        """
        return list(self.default_prompts.keys())

    def get_prompt_info(self, prompt_type: str = None) -> Dict:
        """
        Get information about prompts

        Args:
            prompt_type: Specific prompt type, or None for all

        Returns:
            Dictionary with prompt information
        """
        if prompt_type:
            if prompt_type in self.default_prompts:
                return {
                    'type': prompt_type,
                    'content': self.default_prompts[prompt_type],
                    'length': len(self.default_prompts[prompt_type])
                }
            else:
                return {'error': f'Prompt type "{prompt_type}" not found'}
        else:
            return {
                'available_prompts': self.list_available_prompts(),
                'settings': {
                    'optimization_enabled': self.enable_optimization,
                    'max_length': self.max_prompt_length,
                    'language': self.prompt_language
                }
            }

    def create_custom_prompt(self, base_type: str = 'default', modifications: Dict = None) -> str:
        """
        Create a custom prompt based on existing type with modifications

        Args:
            base_type: Base prompt type to modify
            modifications: Dictionary of modifications to apply

        Returns:
            Custom prompt string
        """
        if base_type not in self.default_prompts:
            base_type = 'default'

        prompt = self.default_prompts[base_type]

        if modifications:
            # Apply intensity modifications
            if 'intensity' in modifications:
                intensity = modifications['intensity'].lower()
                if intensity == 'conservative':
                    prompt = prompt.replace('Remove', 'Carefully remove')
                elif intensity == 'aggressive':
                    prompt = prompt.replace('Remove', 'Aggressively remove')

            # Apply focus modifications
            if 'focus' in modifications:
                focus = modifications['focus']
                prompt = prompt.replace('objects', f'objects, especially {focus}')

        return prompt

# Global instance for easy access
prompt_processor = PromptProcessor()

if __name__ == "__main__":
    # Example usage
    processor = PromptProcessor()

    print("\n📝 Available Prompts:")
    for prompt_type in processor.list_available_prompts():
        info = processor.get_prompt_info(prompt_type)
        print(f"  {prompt_type}: {info['length']} chars")

    print("\n🔧 Prompt Examples:")
    print("Default:", processor.get_prompt('default')[:100] + "...")
    print("Conservative:", processor.get_prompt('conservative')[:100] + "...")

    print("\n🎯 Template Processing:")
    custom = processor.get_prompt('custom_template', image_description='a beautiful sunset over mountains')
    print("Custom:", custom)