# llm_client.py
import os
import json
import re
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Configure model
        self.generation_config = {
            "temperature": 0.3,  # Lower for more consistent output
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        # Use Gemini 1.5 Pro or 2.0 Flash
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",  # or "gemini-1.5-pro"
            generation_config=self.generation_config,
        )
    
    def call_gemini(self, prompt, system_instruction=None, return_json=True):
        """Make API call to Gemini"""
        try:
            # Create model with system instruction if provided
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    generation_config=self.generation_config,
                    system_instruction=system_instruction
                )
            else:
                model = self.model

            response = model.generate_content(prompt)

            # Parse JSON if requested
            if return_json:
                return self.extract_json(response.text)
            return response.text

        except Exception as e:
            print(f"API Error: {e}")
            raise
    
    def extract_json(self, text):
        """Extract JSON from Gemini's response, handling markdown code blocks"""
        # First, try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try any code block
        json_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find the largest JSON object in the text (handles extra text before/after)
        # Look for balanced braces
        stack = []
        start_idx = None
        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        # Found complete JSON object
                        try:
                            json_str = text[start_idx:i+1]
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            # Keep trying
                            start_idx = None

        # Last resort: try simple regex
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Save problematic response for debugging
        debug_file = 'output/debug/failed_json_extraction.txt'
        Path('output/debug').mkdir(parents=True, exist_ok=True)
        with open(debug_file, 'w') as f:
            f.write(f"Failed to extract JSON from response:\n\n{text}\n")

        raise ValueError(f"Could not extract valid JSON from response. Debug saved to: {debug_file}")
    
    def call_and_parse(self, prompt, response_model: BaseModel, system_instruction=None, max_retries=3):
        """Call Gemini and parse response into Pydantic model with retries"""
        for attempt in range(max_retries):
            try:
                # Get response from Gemini
                response_text = self.call_gemini(prompt, system_instruction)
                
                # Extract JSON
                json_data = self.extract_json(response_text)
                
                # Validate with Pydantic
                validated_response = response_model.model_validate(json_data)
                
                return validated_response
            
            except ValidationError as e:
                print(f"Validation error on attempt {attempt + 1}/{max_retries}:")
                print(e)
                
                if attempt == max_retries - 1:
                    # Last attempt failed, save debug info
                    with open(f"error_response_attempt_{attempt + 1}.txt", "w") as f:
                        f.write(response_text)
                    raise
                
                # Add guidance to prompt for next retry
                prompt += f"\n\nPrevious attempt had validation errors. Please ensure the JSON strictly follows this structure:\n{response_model.model_json_schema()}"
            
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise
        
        raise RuntimeError("Failed to get valid response after all retries")


# Test the client
if __name__ == "__main__":
    client = GeminiClient()
    
    # Simple test
    response = client.call_gemini(
        prompt='Respond with JSON: {"message": "Hello from Gemini!", "status": "working"}',
        return_json=True
    )
    print("Parsed JSON response:")
    print(response)