from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from search import search_assets, logger

app = FastAPI(title="Asset Search API", version="1.0.0")

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class SearchResponse(BaseModel):
    assets: list
    total_cost_usd: float
    total_tokens: int
   # search_timestamp: str

@app.post("/search", response_model=SearchResponse)
async def search_assets_endpoint(request: SearchRequest):
    """Search digital assets by text query."""
    try:
        assets, cost, tokens = search_assets(request.query, request.top_k)
        return SearchResponse(
            assets=assets,
            total_cost_usd=cost,
            total_tokens=tokens,
            #search_timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"API search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
