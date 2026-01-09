import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import re
from typing import List, Dict

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
APP_BASE_URL = os.getenv("APP_BASE_URL")

def search_assets(query_text: str, top_k: int = 10) -> tuple[list[dict], float, int]:
    """
    Improved hybrid search with proper text matching.
    """
    try:
        # 1. USE CORRECT MODEL (match your index!)
        embedding_response = openai_client.embeddings.create(
            input=query_text,
            model="text-embedding-3-small"  # Match your index model!
        )
        query_embedding = embedding_response.data[0].embedding
        total_tokens = embedding_response.usage.total_tokens
        cost_per_token = 0.00000002
        total_cost = total_tokens * cost_per_token
        
        # 2. RETRIEVE (over-fetch for reranking)
        fetch_k = top_k * 3
        results = index.query(
            namespace=NAMESPACE,
            vector=query_embedding,
            top_k=fetch_k,
            include_metadata=True
        )
        
        # 3. PREPARE QUERY TOKENS (improved tokenization)
        query_words = set(query_text.lower().split())
        
        assets = []
        for match in results.matches:
            md = match.metadata or {}
            vector_score = float(match.score)
            
            # 4. FUZZY TAG MATCHING (word overlap, not exact match)
            tags_text = " ".join(md.get("tags", [])).lower()
            tag_words = set(tags_text.split())
            tag_score = len(query_words & tag_words) / max(len(query_words), 1)
            
            # 5. FUZZY DESCRIPTION MATCHING
            desc_words = set(md.get("description", "").lower().split())
            desc_score = len(query_words & desc_words) / max(len(query_words), 1)
            
            # 6. COMBINED TEXT MATCHING (now actually used!)
            combined_words = set(md.get("combined_text", "").lower().split())
            combined_score = len(query_words & combined_words) / max(len(query_words), 1)
            
            # 7. SCENE TYPE MATCHING
            scene_score = 0.0
            if md.get("scene_type"):
                scene_words = set(md["scene_type"].lower().split())
                scene_score = len(query_words & scene_words) / max(len(query_words), 1)
            
            # 8. BALANCED SCORING (vector-first, metadata-second)
            final_score = (
                0.70 * vector_score +      # Primary: semantic similarity
                0.12 * tag_score +          # Secondary: tag relevance
                0.10 * combined_score +     # Tertiary: full-text match
                0.05 * desc_score +         # Quaternary: description
                0.03 * scene_score          # Quinary: scene type
            )
            
            asset = {
                "id": match.id,
                "asset_url": APP_BASE_URL + match.id,
                "image_url": md.get("image_url"),
                "vector_score": vector_score,
                "final_score": final_score,
                "tags": md.get("tags", []),
                "description": md.get("description", ""),
                "why": {
                    "tag_score": round(tag_score, 2),
                    "desc_score": round(desc_score, 2),
                    "combined_score": round(combined_score, 2),
                    "scene_score": round(scene_score, 2),
                    "query_words": list(query_words),
                    "tag_hits": list(query_words & tag_words)
                }
            }
            assets.append(asset)
        
        # 9. SORT AND TRIM
        assets.sort(key=lambda x: x["final_score"], reverse=True)
        assets = assets[:top_k]
        tag_hits = query_words & tag_words 
        logger.info(
            f"SEARCH '{query_text}' | "
            f"results={len(assets)} | "
            f"top_score={assets[0]['final_score']:.3f} if assets else 'n/a'"
        )
        logger.info(
            f"Asset {match.id}: "
            f"vector={vector_score:.3f} "
            f"tag={tag_score:.3f} (hits: {tag_hits}) "
            f"desc={desc_score:.3f} "
            f"final={final_score:.3f}"
        )
        
        return assets, total_cost, total_tokens
        
    except Exception as e:
        logger.exception(f"Search failed for '{query_text}'")
        raise

        