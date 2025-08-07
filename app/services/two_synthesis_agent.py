import google.generativeai as genai
from typing import List, AsyncGenerator
from app.core.config import settings

# Configure the Gemini client with your API key
genai.configure(api_key=settings.GOOGLE_API_KEY)

class SynthesisAgent:
    """
    An advanced agent that uses a Chain of Thought process to synthesize, critique,
    and refine an answer in a single, streaming API call, optimized for brevity.
    """
    def __init__(self):
        """
        Initializes the SynthesisAgent, setting up the Gemini model for text generation.
        """
        self.model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

    async def synthesize_final_answer(self, original_question: str, context_chunks: List[str]) -> str:
        """
        Generates a final, high-quality, and CONCISE answer by performing an internal
        synthesis, critique, and refinement process, then returns the complete answer.

        Args:
            original_question: The user's original, complete question.
            context_chunks: A list of relevant text chunks from the document.

        Returns:
            The complete, refined answer text.
        """
        context_str = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""
You are a precise and factual writing assistant. Your primary goal is to answer the user's question as CONCISELY as possible based ONLY on the provided 'Source Text'.

Follow this internal Chain of Thought, but DO NOT output the thought process. Your final output should only be the answer text.

1.  **Synthesize**: First, find the most direct and relevant information in the 'Source Text' to answer the 'User's Original Question'.
2.  **Critique for Brevity**: Second, critically review your answer. Ask yourself: "Is this the shortest possible answer that is still accurate and complete? Have I included unnecessary details, examples, or lists? Can I say this in fewer words?"
3.  **Refine for Conciseness**: Finally, rewrite the answer to be as brief and to-the-point as possible. Summarize the key information into a single, clear sentence or two.

**CRITICAL RULES:**
-   **BE BRIEF**: Do not provide long, multi-paragraph explanations.
-   **SUMMARIZE**: Do not list out all details. Summarize the key finding.
-   **DIRECT ANSWERS**: Get straight to the point.

---
Source Text:
{context_str}
---
User's Original Question:
"{original_question}"
---
Final, Concise Answer:
"""

        try:
            # Use generate_content without streaming to get the complete answer
            response = await self.model.generate_content_async(prompt)
            return response.text
            
        except Exception as e:
            print(f"Error in Gemini Synthesis Agent: {e}")
            return "[An error occurred while generating the final answer.]"

    # async def synthesize_final_answer_stream(self, original_question: str, context_chunks: List[str]) -> AsyncGenerator[str, None]:
    #     """
    #     Generates a final, high-quality answer by performing an internal
    #     synthesis, critique, and refinement process, then streams the result.
    # 
    #     Args:
    #         original_question: The user's original, complete question.
    #         context_chunks: A list of relevant text chunks from the document.
    # 
    #     Yields:
    #         Chunks of the final, refined answer text as they are generated.
    #     """
    #     context_str = "\n\n---\n\n".join(context_chunks)
    #     
    #     prompt = f"""
    # You are a precise and factual writing assistant. Your task is to generate a final, high-quality answer to the user's question based ONLY on the provided 'Source Text'.
    # 
    # Follow this internal Chain of Thought to construct your response, but DO NOT output the thought process itself. Your final output should only be the answer text.
    # 
    # 1.  **Synthesize a Draft**: First, carefully review the 'Source Text' and the 'User's Original Question'. Compose a comprehensive draft answer that directly addresses the question using only the information provided in the source text.
    # 2.  **Critique the Draft**: Second, critically review your own draft. Ask yourself: "Is every single statement I wrote directly supported by the source text? Did I miss any key details needed to answer the original question completely? Is the answer clear and concise?"
    # 3.  **Refine and Finalize**: Finally, based on your internal critique, revise the draft to correct any inaccuracies, add any missing information, and improve its clarity.
    # 
    # Your final output to the user should be ONLY the refined, polished, and factually-grounded answer. Do not include any of your internal thoughts, drafts, or critiques.
    # 
    # ---
    # Source Text:
    # {context_str}
    # ---
    # User's Original Question:
    # "{original_question}"
    # ---
    # Final Answer:
    # """
    # 
    #     try:
    #         # Use generate_content with stream=True to get the final answer token-by-token
    #         response_stream = await self.model.generate_content_async(prompt, stream=True)
    #         
    #         async for chunk in response_stream:
    #             if chunk.text:
    #                 yield chunk.text
    #             
    #     except Exception as e:
    #         print(f"Error in Gemini Synthesis Agent (Streaming): {e}")
    #         yield "[An error occurred while generating the final answer.]"

