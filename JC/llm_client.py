# llm_client.py
import os
import json
import re
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
            model_name="gemini-2.0-flash-exp",  # or "gemini-1.5-pro"
            generation_config=self.generation_config,
        )
    
    def call_gemini(self, prompt, system_instruction=None):
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
        
        # Try to find any JSON object in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError("Could not extract valid JSON from response")
    
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
    )
    print("Raw response:")
    print(response)
    print("\nExtracted JSON:")
    print(client.extract_json(response))