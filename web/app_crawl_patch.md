# Crawler admin

After deploy, `/admin` shows Crawler section with:
- ON/OFF switches (stored in Neon via DATABASE_URL)
- Crawl URL
- Search & crawl (DuckDuckGo HTML results → fetch pages)
- Random crawl

Docs persist in Postgres table `crawl_docs` when DATABASE_URL is set.
