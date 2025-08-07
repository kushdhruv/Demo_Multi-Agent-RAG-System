import google.generativeai as genai
from typing import List, Dict
from app.core.config import settings

# Configure the Gemini client with your API key
genai.configure(api_key=settings.GOOGLE_API_KEY)

class CombinedAgent:
    """
    A highly efficient agent that processes all questions in a single API call
    to minimize processing time and includes retry logic for better accuracy.
    """
    
    def __init__(self):
        """
        Initializes the CombinedAgent with optimized Gemini model configuration.
        """
        generation_config = genai.GenerationConfig(
            temperature=0.1,  # Low temperature for consistent, factual answers
            top_p=0.8,
            top_k=40
        )
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-05-20",
            generation_config=generation_config
        )

    async def process_all_questions_with_retry(self, questions: List[str], context_chunks: List[str], max_retries: int = 2) -> List[str]:
        """
        Processes all questions ideally in a single API call. If some answers are
        missing/insufficient, performs ONE more call only for the unanswered
        questions (total tries capped at 2).
        
        Args:
            questions: List of questions to answer
            context_chunks: Relevant text chunks from the document
            max_retries: Maximum number of total tries (default: 2)
            
        Returns:
            List of answers corresponding to the questions (same order)
        """
        context_str = "\n\n---\n\n".join(context_chunks)
        
        # Build first-pass prompt for all questions
        questions_text = "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions)])
        prompt_all = f"""
You are a precise and factual insurance policy assistant. Your task is to answer multiple questions about the National Parivar Mediclaim Plus Policy based ONLY on the provided 'Source Text'.

CRITICAL RULES:
- Answer each question CONCISELY (1-2 sentences maximum)
- Use ONLY information from the Source Text
- Be direct and factual
- If information is not found, respond exactly: "Information not found in the policy document"
- Number your answers to match the question numbers

Source Text:
{context_str}

Questions to Answer:
{questions_text}

Instructions:
Provide your answers in this exact format:
1. [Answer to question 1]
2. [Answer to question 2]
3. [Answer to question 3]
... and so on for all questions.

Final Answers:
"""

        # First attempt: answer all
        print("🔄 Processing all questions (attempt 1)")
        try:
            response = await self.model.generate_content_async(prompt_all)
            response_text = response.text.strip()
            answers_map = self._parse_numbered_answers_map(response_text)
        except Exception as e:
            print(f"❌ Error in attempt 1: {e}")
            answers_map = {}
        
        # Build initial results; None for missing
        results: List[str] = [answers_map.get(i + 1) for i in range(len(questions))]
        missing_indices = [i for i, ans in enumerate(results) if not ans]
        
        # If everything answered or retries not allowed, finalize
        if not missing_indices or max_retries <= 1:
            return [ans if ans else "Information not found in the policy document" for ans in results]
        
        # Second attempt: only retry unanswered questions
        print(f"🔁 Retrying only unanswered questions (attempt 2) for indices: {[i+1 for i in missing_indices]}")
        missing_questions_text = "\n".join([f"{(i+1)}. {questions[i]}" for i in missing_indices])
        prompt_missing = f"""
You are a precise and factual insurance policy assistant. Answer ONLY the listed question numbers based ONLY on the provided 'Source Text'.

CRITICAL RULES:
- Answer CONCISELY (1-2 sentences maximum)
- Use ONLY information from the Source Text
- Be direct and factual
- If information is not found, respond exactly: "Information not found in the policy document"
- Number your answers to match the ORIGINAL question numbers provided below

Source Text:
{context_str}

Questions to Answer (use these original numbers in your answers):
{missing_questions_text}

Instructions:
Provide your answers in this exact format using the original numbers:
{''.join([str(i+1)+'. [Answer]\n' for i in missing_indices])}

Final Answers:
"""
        try:
            response2 = await self.model.generate_content_async(prompt_missing)
            response2_text = response2.text.strip()
            retry_map = self._parse_numbered_answers_map(response2_text)
            # Fill in missing only if provided
            for idx in missing_indices:
                ans = retry_map.get(idx + 1)
                if ans:
                    results[idx] = ans
        except Exception as e:
            print(f"❌ Error in attempt 2: {e}")
            # leave missing as None
        
        # Finalize with placeholders for anything still missing
        return [ans if ans else "Information not found in the policy document" for ans in results]

    def _parse_numbered_answers_map(self, response_text: str) -> Dict[int, str]:
        """
        Parses numbered answers like "1. ...", "2. ..." into a map of index->answer.
        Tolerates lines with extra whitespace. Only captures leading integer index.
        """
        lines = response_text.split('\n')
        answers: Dict[int, str] = {}
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            # Find patterns like "12. answer..."
            if line[0].isdigit():
                # Extract leading integer
                num_str = ''
                j = 0
                while j < len(line) and line[j].isdigit():
                    num_str += line[j]
                    j += 1
                # Expect a dot after number
                if j < len(line) and line[j] == '.':
                    try:
                        idx = int(num_str)
                        ans = line[j+1:].strip()
                        if ans:
                            answers[idx] = ans
                    except ValueError:
                        continue
        return answers

    async def process_single_question_with_context(self, question: str, context_chunks: List[str]) -> str:
        """
        Processes a single question with context (for fallback scenarios).
        
        Args:
            question: Single question to answer
            context_chunks: Relevant text chunks from the document
            
        Returns:
            Answer to the question
        """
        context_str = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""
You are a precise and factual insurance policy assistant. Answer the following question based ONLY on the provided 'Source Text'.

CRITICAL RULES:
- Answer CONCISELY (1-2 sentences maximum)
- Use ONLY information from the Source Text
- Be direct and factual
- If information is not found, respond exactly: "Information not found in the policy document"

Source Text:
{context_str}

Question:
{question}

Answer:
"""

        try:
            response = await self.model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error processing single question: {e}")
            return "Information not found in the policy document"
