from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from app.schemas.models import RunRequest, IngestRequest
from app.core.security import verify_token
import asyncio
import json
from typing import List

# Import your agent classes
from app.services.three_retrieval_service import RetrievalService
from app.services.one_planning_agent import PlanningAgent
from app.services.two_synthesis_agent import SynthesisAgent
from app.services.combined_agent import CombinedAgent

# Create an API router
router = APIRouter()

# --- Singleton services with persistent state ---
class ServiceManager:
    """Singleton manager for all services to maintain state across requests."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.retrieval_service = None
            self.planning_agent = None
            self.synthesis_agent = None
            self.combined_agent = None
            self.question_cache = {}
            self.policy_preloaded = False
            self._initialized = True
    
    def initialize_services(self):
        """Initialize services only once."""
        if self.retrieval_service is None:
            print("First request received. Initializing all services...")
            self.retrieval_service = RetrievalService()
            self.planning_agent = PlanningAgent()
            self.synthesis_agent = SynthesisAgent()
            self.combined_agent = CombinedAgent()
            print("All services initialized.")

# Global service manager instance
service_manager = ServiceManager()

# --- Main JSON Q&A Endpoint ---
@router.post("/hackrx/run")
async def run_submission(request: RunRequest, _token: str = Depends(verify_token)):
    """
    Orchestrates the highly optimized Q&A workflow using combined agent for single API call processing.
    Uses the existing Pinecone index if available; otherwise recreates it from the local policy file.
    """
    service_manager.initialize_services()

    policy_path = "./data/policy.pdf"

    # Ensure an index is available; if missing, recreate from local policy file
    if service_manager.retrieval_service.index is None:
        try:
            print(f"🆕 No Pinecone index found. Creating from local policy file: {policy_path}")
            await asyncio.to_thread(service_manager.retrieval_service.ingest_and_process_pdf, policy_path)
            print(f"✅ Successfully created Pinecone index from: {policy_path}")
            service_manager.policy_preloaded = True
        except Exception as e:
            print(f"❌ Failed to create Pinecone index from local file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to (re)create Pinecone index from local policy file: {e}"
            )
    else:
        print("📊 Using existing Pinecone index - skipping ingestion step")
        service_manager.policy_preloaded = True

    # Always clear cache for fresh answers per request batch
    service_manager.question_cache.clear()

    # Process all questions using the combined agent with retry logic
    try:
        print(f"🚀 Processing {len(request.questions)} questions with combined agent...")
        
        # Get all relevant context chunks for all questions
        all_context_chunks = await get_comprehensive_context(request.questions)
        
        # Process all questions in a single API call
        answers = await service_manager.combined_agent.process_all_questions_with_retry(
            questions=request.questions,
            context_chunks=all_context_chunks,
            max_retries=2
        )
        
        print(f"✅ Successfully processed all {len(answers)} questions")

    except Exception as e:
        print(f"Error processing questions: {e}")
        # Fallback: process questions individually
        answers = []
        for i, question in enumerate(request.questions):
            try:
                print(f"Processing question {i+1}/{len(request.questions)}: {question}")
                answer = await run_single_question_pipeline(question)
                answers.append(answer)
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                error_message = f"An error occurred while processing the question: {question}"
                answers.append(error_message)

    # Return JSON response with answers array
    return JSONResponse(content={"answers": answers})

async def get_comprehensive_context(questions: List[str]) -> List[str]:
    """
    Gets comprehensive context chunks for all questions to ensure good coverage.
    """
    print("🔍 Gathering comprehensive context for all questions...")
    
    all_context_chunks = set()
    
    # Use a higher top_k for comprehensive context gathering
    for question in questions:
        try:
            # Get context for each question
            context = await asyncio.to_thread(
                service_manager.retrieval_service.search_and_rerank, 
                question, 
                top_k_retrieval=30  # Higher retrieval for better coverage
            )
            all_context_chunks.update(context)
        except Exception as e:
            print(f"Warning: Could not get context for question '{question}': {e}")
    
    context_list = list(all_context_chunks)
    print(f"📚 Retrieved {len(context_list)} unique context chunks for all questions")
    
    return context_list

async def run_single_question_pipeline(question: str) -> str:
    """
    Runs the optimized 2-agent pipeline for a single question and returns the complete answer.
    (Fallback method for individual question processing)
    """
    print(f"\n--- Processing Question: {question} ---")
    
    if question in service_manager.question_cache:
        print("Found in cache.")
        return service_manager.question_cache[question]

    # 1. Planning Agent: Decomposes and generates hypothetical answers in one call
    research_plan = service_manager.planning_agent.plan_and_research(question)
    print(f"Research plan generated: {len(research_plan)} sub-question(s)")

    # 2. Concurrent Context Retrieval with retry logic
    all_context_chunks = set()
    max_retries = 2
    
    for attempt in range(max_retries):
        async def get_context_for_sub_q(sub_q: str, hypothetical_answers: list):
            local_chunks = set()
            # Increase top_k on retry attempts
            top_k = 60 if attempt > 0 else (40 if len(research_plan) > 1 else 20)
            for ha in hypothetical_answers:
                context = await asyncio.to_thread(service_manager.retrieval_service.search_and_rerank, ha, top_k_retrieval=top_k)
                local_chunks.update(context)
            return local_chunks

        tasks = [get_context_for_sub_q(sub_q, hypos) for sub_q, hypos in research_plan.items()]
        results = await asyncio.gather(*tasks)
        
        # Collect all chunks from this attempt
        attempt_chunks = set()
        for chunk_set in results:
            attempt_chunks.update(chunk_set)
        
        all_context_chunks.update(attempt_chunks)
        
        print(f"Retrieved {len(attempt_chunks)} unique context chunks (attempt {attempt + 1}/{max_retries}).")
        
        # If we found chunks, break out of retry loop
        if attempt_chunks:
            break
        elif attempt < max_retries - 1:
            print(f"⚠️ No relevant chunks found on attempt {attempt + 1}, retrying with more chunks...")
    
    if not all_context_chunks:
        print("⚠️ Warning: No relevant chunks found after all retry attempts.")
        return "I couldn't find specific information about this in the policy document. Please try rephrasing your question or contact customer support for detailed policy information."

    print(f"Retrieved {len(all_context_chunks)} unique context chunks.")

    # 3. Synthesis Agent: Generates the final answer
    final_answer = await service_manager.synthesis_agent.synthesize_final_answer(question, list(all_context_chunks))
    
    # Cache the answer
    service_manager.question_cache[question] = final_answer
    
    return final_answer
