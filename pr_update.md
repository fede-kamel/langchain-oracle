## Live Examples & Demos

**Gist with tested examples**: https://gist.github.com/fede-kamel/f737dce96a7f8244e9c4ba94d7122eb8

The gist contains fully tested, working examples:

### ✅ Validated Examples (Tested 2026-03-03)

1. **`deep_research_adb_datastore.py`** - ADB vector search
   - Tested with 980 documents
   - Autonomous semantic search + synthesis
   - Model: google.gemini-2.5-pro

2. **`deep_research_agent_demo.py`** - OpenSearch integration  
   - Tested with SRE diagnostic patterns
   - Agent made 12 autonomous tool calls
   - Retrieved 6 diagnostic patterns

3. **`deep_research_oci_storage.py`** - OCI Object Storage
   - Creates tools for bucket operations
   - Searches medical/legal datasets

### Sample Outputs Included

- `SAMPLE_NEUROLOGY_RESEARCH.md` - 58KB comprehensive medical case analysis (18 pages, 13 citations)
- `SAMPLE_LEGAL_COMPLIANCE_RESEARCH.md` - 30KB contract analysis
- `SAMPLE_SRE_INVESTIGATION.md` - 12KB SRE diagnostic research
- `UPDATE_GIST_NOTES.md` - Complete testing documentation

### Test Results

**ADB Deep Research:**
```
Query: "Find documents about leukocytosis diagnosis and treatment"
Result: Agent searched 980 documents and synthesized findings about 
        WBC counts and leukocytosis diagnosis.
```

**OpenSearch Deep Research:**
```
Query: "Analyze SRE diagnostic patterns for connection_exhaustion, memory, timeout"
Tools Called: stats(), keyword_search() x3, get_document() x6
Result: Retrieved and analyzed 6 diagnostic patterns, generated 
        comprehensive technical summary with tool recommendations.
```

All examples use the correct API (`datastore_description` parameter) and include complete dependency requirements.
