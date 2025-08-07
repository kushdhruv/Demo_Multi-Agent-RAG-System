import json
import google.generativeai as genai
from typing import List, Dict
from app.core.config import settings

# Configure the Gemini client with your API key
genai.configure(api_key=settings.GOOGLE_API_KEY)

class PlanningAgent:
    """
    An advanced agent that combines decomposition and hypothetical answer generation
    into a single, efficient Chain of Thought operation.
    """
    def __init__(self):
        """
        Initializes the PlanningAgent, setting up the Gemini model for JSON output.
        """
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
        )
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-05-20",
            generation_config=generation_config
        )

    def plan_and_research(self, question: str) -> Dict[str, List[str]]:
        """
        Decomposes a question and generates hypothetical answers for each sub-question.

        Args:
            question: The user's original, potentially complex question.

        Returns:
            A dictionary where keys are sub-questions and values are lists of
            hypothetical answers for that sub-question.
        """
        safe_question = json.dumps(question)

        prompt = f"""
You are a master of logical reasoning and query planning. Your task is to analyze the user's question and create a research plan.

Follow these steps in your reasoning process:
1.  **Decomposition**: First, break down the user's question into a series of simpler, self-contained sub-questions. If the question is already simple, treat it as a single sub-question.
2.  **Hypothetical Generation**: For each sub-question you identified, generate three different and detailed hypothetical answers. These answers should be plausible and information-rich to be used for a vector search.

Your final output MUST be a single, valid JSON object. The keys of the object should be the sub-questions you generated, and the value for each key should be an array of the three hypothetical answers for that sub-question.

User Question: {safe_question}
"""

        try:
            response = self.model.generate_content(prompt)
            response_data = json.loads(response.text)
            
            # Fallback if the model fails to produce a valid structure
            if not isinstance(response_data, dict) or not response_data:
                print(f"Warning: Planning agent returned unexpected structure. Falling back.")
                return {question: [question]} # Simple fallback

            return response_data

        except Exception as e:
            print(f"Error in Gemini Planning Agent: {e}")
            # If the LLM call fails, fall back to a simple structure.
            return {question: [question]}
