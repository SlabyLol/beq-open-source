# Beq Crawl Bots + Search Tool

## Answer order
1. `configs/*.sbe` (exact FAQ)
2. Math calculator
3. **Search tool** — crawl store → Wikipedia → DuckDuckGo
4. Neural model

## APIs

```bash
# Public search
curl "https://beq.onrender.com/api/search?q=What+is+PyTorch"

# Admin: crawl a public page (must be logged in as admin)
curl -X POST "https://beq.onrender.com/api/crawl" -F "url=https://en.wikipedia.org/wiki/Python_(programming_language)"

# List stored docs (admin)
curl "https://beq.onrender.com/api/crawl/docs"
```

## Storage
- `data/crawl_store/index.jsonl` — crawled pages (URL + text)
- `data/crawl_corpus.txt` — append-only text useful for later training

## Limits
- Public http/https only
- ~80KB text per page
- Max ~200 docs in index
- No API keys required for Wikipedia / DuckDuckGo
