import google.generativeai as genai
from google.api_core import retry
import os
#
import logging

logger = logging.getLogger(__name__)

class ChatModel():
    def __init__(self, model):
        self.model = model
        
        # Use Gemini API key from config
        if "gemini" in self.model.lower():
            #api_key = Config.GEMINI_API_KEY
            api_key = "AIzaSyDKmb2Vz1LUisaH9nDt_VmQDlVEjmVQYO4"
            if not api_key:
                raise ValueError("GEMINI_API_KEY not configured")
            
            genai.configure(api_key=api_key)
            
            try:
                self.generator = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
            except Exception as e:
                logger.error(f"Error initializing Gemini: {str(e)}")
                raise

    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def chat(self, system_prompt, user_prompt):
        """Single method for chat using Gemini"""
        try:
            full_prompt = f"""System: {system_prompt}

User Request: {user_prompt}

Please provide a detailed response following these formatting guidelines:

1. Start with a clear summary of environmental conditions using bullet points and emojis
   Example:
   🌡️ Temperature: XX°C
   💧 Soil Moisture: XX m³/m³
   etc.

2. Structure each section with:
   - Clear emoji headers
   - Bulleted lists for easy reading
   - Spacing between sections
   - Important points highlighted with emojis
   - Tables where appropriate

3. Use these sections with emojis:
   🌱 PLANTING TIMELINE
   💧 IRRIGATION & SOIL MANAGEMENT
   🌿 FERTILIZATION STRATEGY
   🌍 ENVIRONMENTAL CONSIDERATIONS
   🔄 MAINTENANCE SCHEDULE

4. Format numbers and measurements clearly:
   - Use proper units
   - Round decimals appropriately
   - Present ranges with clear min-max values

5. End with a quick-reference summary of key points

Please maintain all the environmental data values exactly as provided, but present them in a more visually appealing and easy-to-read format."""

            response = self.generator.generate_content(
                full_prompt,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 4096,
                }
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."
