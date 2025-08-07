#!/usr/bin/env python3
"""
Preload Policy Document Script
This script preloads the policy document into the vector database
so it's ready for queries without processing it every time.
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.three_retrieval_service import RetrievalService

async def preload_policy_document():
    """
    Preloads the policy document into the vector database.
    """
    print("🚀 Starting policy document preload...")
    
    # Initialize the retrieval service
    retrieval_service = RetrievalService()
    
    # Path to the policy document
    policy_path = "./data/policy.pdf"
    
    if not os.path.exists(policy_path):
        print(f"❌ Error: Policy document not found at {policy_path}")
        print("Please ensure the policy.pdf file is in the ./data directory")
        return False
    
    try:
        print(f"📄 Processing policy document: {policy_path}")
        
        # Process the PDF and upload to vector database
        await asyncio.to_thread(retrieval_service.ingest_and_process_pdf, policy_path)
        
        print("✅ Policy document successfully preloaded into vector database!")
        print("📊 Document is now ready for queries without processing delay.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error preloading policy document: {e}")
        return False

def main():
    """
    Main function to run the preload script.
    """
    print("=" * 60)
    print("🔧 POLICY DOCUMENT PRELOAD SCRIPT")
    print("=" * 60)
    
    # Run the preload
    success = asyncio.run(preload_policy_document())
    
    if success:
        print("\n🎉 Preload completed successfully!")
        print("💡 You can now use the API without document processing delays.")
        print("📝 Use './data/policy.pdf' as the documents parameter in your API calls.")
    else:
        print("\n💥 Preload failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 