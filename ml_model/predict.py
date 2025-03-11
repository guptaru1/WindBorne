import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class AgriculturalLLM:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def get_agricultural_insight(self, query, weather_data=None):
        # Combine query with weather data if available
        input_text = query
        if weather_data:
            input_text = f"Weather: {weather_data}\nQuery: {query}"

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

        # Map prediction to insight
        severity_mapping = {
            0: "Low",
            1: "Medium",
            2: "High"
        }
        severity = severity_mapping[predicted_class]

        # Generate response based on prediction
        response = self._generate_response(query, severity, confidence, weather_data)
        return response

    def _generate_response(self, query, severity, confidence, weather_data):
        """Generate a detailed response based on the model's prediction"""
        response_templates = {
            "Low": {
                "risk": "Current conditions indicate low risk. Basic precautions should be sufficient.",
                "action": "Continue regular monitoring and maintenance."
            },
            "Medium": {
                "risk": "Moderate risk detected. Enhanced monitoring recommended.",
                "action": "Consider implementing preventive measures and increase monitoring frequency."
            },
            "High": {
                "risk": "High risk alert! Immediate attention required.",
                "action": "Implement protective measures immediately and prepare for potential interventions."
            }
        }

        template = response_templates[severity]
        
        response = f"""Analysis Results:
Risk Level: {severity} (Confidence: {confidence:.2%})

Current Assessment:
{template['risk']}

Recommended Actions:
{template['action']}

"""
        if weather_data:
            response += f"\nWeather Conditions:\n{weather_data}"

        return response 