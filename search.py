import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asset_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
NAMESPACE = os.getenv("PINECONE_NAMESPACE")

def search_assets(query_text: str, top_k: int = 10) -> tuple[List[Dict], float, int]:
    """
    Search assets with token/cost tracking.
    
    Returns: (assets, total_cost_usd, total_tokens)
    """
    #start_time = datetime.now()
    
    try:
        # Generate embedding
        embedding_response = openai_client.embeddings.create(
            input=query_text,
            model="text-embedding-3-small"
        )
        query_embedding = embedding_response.data[0].embedding
        input_tokens = embedding_response.usage.prompt_tokens
        total_tokens = embedding_response.usage.total_tokens
        
        # Calculate cost ($0.02 per 1M tokens for text-embedding-3-small)
        cost_per_token = 0.00000002  # $0.02 / 1M tokens
        total_cost = total_tokens * cost_per_token
        
        # Perform search
        results = index.query(
            namespace=NAMESPACE,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract assets
        assets = []
        for match in results.matches:
            asset = {
                "id":  match.id,
                "asset_url": "https://qa.sparkfivetest.com/main/assets/" + match.id,
                "score": float(match.score),
                "image_url": match.metadata.get("image_url"),
                "description": match.metadata.get("description", ""),
                "tags": match.metadata.get("tags", [])
            }
            assets.append(asset)
        
        #duration = (datetime.now() - start_time).total_seconds()
        
        # Log search details
        logger.info(
            f"SEARCH: '{query_text}' | "
            f"tokens: {total_tokens} | "
            f"cost: ${total_cost:.6f} | "
            f"results: {len(assets)} | "
           #f"duration: {duration:.2f}s"
        )
        
        return assets, total_cost, total_tokens
        
    except Exception as e:
        logger.error(f"Search failed for '{query_text}': {str(e)}")
        raise
