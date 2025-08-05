# AI Tool Usage Extraction Pipeline

A comprehensive pipeline for extracting AI tool usage mentions from Swiss job postings, implementing the Hampole et al. methodology.

## Quick Start

### 1. Setup
```bash
cd Code/
pip install -r requirements.txt
```

### 2. Configuration
Edit `Code/config.env` with your credentials:
```env
DB_PASSWORD=your_actual_password
OPENAI_API_KEY=your_api_key_here
```

### 3. Run Pipeline
```bash
cd Code/
python3 ai_pipeline.py
```

## Project Structure
```
├── Code/                    # All code and configuration files
│   ├── ai_pipeline.py      # Main pipeline script
│   ├── config.env          # Configuration file
│   ├── requirements.txt    # Dependencies
│   └── JobAds/            # Database utilities
├── data/                   # Generated data files
└── PROJECT_STATUS.md       # Auto-generated documentation
```

## Features
- ✅ **Multi-company data extraction** from PostgreSQL
- ✅ **Multilingual processing** (DE/FR/IT/EN)
- ✅ **Keyword-based filtering** for AI tool mentions
- ✅ **Automated documentation** updates
- 🔄 **LLM processing** (coming soon)

## Documentation
- **PROJECT_STATUS.md** - Auto-generated detailed project status
- **ai_tool_usage_extraction_10_companies.md** - Original methodology guide

## Update Documentation
```bash
cd Code/
python3 update_documentation.py
```

---
*For detailed status and metrics, see [PROJECT_STATUS.md](PROJECT_STATUS.md)*