# Medical Research MCP Agent

A specialized medical research assistant agent that connects to multiple medical and research-focused MCP (Model Context Protocol) servers through Smithery.ai. This agent is designed to help with medical research, clinical information, and healthcare-related queries.

## Capabilities

This agent can:
1. Search medical research papers through PubMed
2. Find clinical trials information
3. Search academic papers
4. Perform medical calculations
5. Search general medical information

## Connected MCP Servers

The agent connects to the following specialized medical and research servers:

1. **PubMed MCP Server** (@JackKuo666/pubmed-mcp-server)
   - Access to PubMed medical research database
   - Available tools:
     - search_pubmed_key_words: Search by keywords
     - search_pubmed_advanced: Advanced search with filters
     - get_pubmed_article_metadata: Get detailed article information
     - download_pubmed_pdf: Download full papers

2. **Clinical Trials MCP Server** (@JackKuo666/clinicaltrials-mcp-server)
   - Access to clinical trials database
   - Available tools:
     - search_clinical_trials_and_save_studies_to_csv: Search and save results
     - get_full_study_details: Get comprehensive trial information
     - get_studies_by_keyword: Search by keywords
     - get_study_statistics: Get trial statistics


3. **Paper Search MCP Server** (@openags/paper-search-mcp)
   - Academic paper search capabilities
   - Available tools:
     - search_arxiv: Search arXiv papers
     - search_pubmed: Search PubMed papers
     - search_biorxiv: Search bioRxiv papers
     - search_google_scholar: Search Google Scholar
     - download_arxiv: Download arXiv papers
     - download_pubmed: Download PubMed papers
     - download_biorxiv: Download bioRxiv papers
     - read_arxiv_paper: Read arXiv papers
     - read_pubmed_paper: Read PubMed papers
     - read_biorxiv_paper: Read bioRxiv papers


4. **MedCalc MCP Server** (@vitaldb/medcalc)
   - Medical calculations and formulas
   - Available tools:
     - egfr_epi_cr_cys: eGFR calculation
     - bp_children: Blood pressure for children
     - bmi_bsa_calculator: BMI and BSA calculation
     - crcl_cockcroft_gault: Creatinine clearance
     - map_calculator: Mean arterial pressure
     - chads2_vasc_score: CHA2DS2-VASc score
     - prevent_cvd_risk: CVD risk assessment
     - corrected_calcium: Calcium correction
     - qtc_calculator: QTc interval
     - wells_pe_criteria: Wells PE criteria
     - ibw_abw_calculator: Ideal/Adjusted body weight
     - pregnancy_calculator: Pregnancy calculations
     - revised_cardiac_risk_index: Cardiac risk assessment
     - child_pugh_score: Child-Pugh score
     - steroid_conversion: Steroid conversion
     - calculate_mme: Morphine milligram equivalents
     - maintenance_fluids: Fluid maintenance
     - corrected_sodium: Sodium correction
     - meld_3: MELD score
     - framingham_risk_score: Framingham risk
     - homa_ir: HOMA-IR calculation


5. **DuckDuckGo MCP Server** (@nickclyde/duckduckgo-mcp-server)
   - General medical information search
   - Available tools:
     - search: Web search
     - fetch_content: Fetch web content

