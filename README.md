# AgeSci

## Autonomous Multi-Agent Research Paper Generation System

AgeSci is a fully autonomous system that generates complete IEEE-format academic research papers from a single topic input. The system orchestrates multiple specialized AI agents through local large language models via Ollama, with each agent responsible for distinct aspects of the research paper creation pipeline: architectural planning, literature research, academic writing, peer review, diagram generation, bibliography management, and LaTeX compilation.

The name "AgeSci" reflects the project's core philosophy: Agents doing Science.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Dependencies](#dependencies)
6. [Usage](#usage)
7. [Configuration](#configuration)
8. [Output Structure](#output-structure)
9. [Agent System](#agent-system)
10. [Citation Pipeline](#citation-pipeline)
11. [Quality Control](#quality-control)
12. [Example Output](#example-output)
13. [Limitations and Considerations](#limitations-and-considerations)
14. [Troubleshooting](#troubleshooting)
15. [Contributing](#contributing)
16. [License](#license)
17. [Citation](#citation)

---

## Overview

Academic paper writing traditionally demands extensive literature review, meticulous structuring, proper citation management, and strict adherence to formatting standards. AgeSci automates this entire workflow by deploying a coordinated team of specialized AI agents that collaborate to produce publication-ready documents.

Given a research topic as input, the system autonomously performs the following operations:

- Designs a comprehensive paper structure with appropriate sections and subsections
- Extracts domain-specific search terms using LLM-powered keyword analysis
- Queries academic databases (arXiv, Semantic Scholar) for relevant citations
- Writes each section with formal academic tone and proper LaTeX formatting
- Conducts iterative peer review with scoring and revision cycles
- Generates TikZ diagrams for methodology and architecture visualization
- Compiles the final LaTeX document with complete bibliography
- Produces a PDF output ready for review or submission

All citations are sourced from real academic papers through the arXiv and Semantic Scholar APIs, ensuring verifiability and academic integrity.

---

## Key Features

### Real Citation Integration

Unlike generative systems that fabricate references, AgeSci retrieves actual papers from established academic databases:

- **arXiv API**: Direct access to preprints across physics, mathematics, computer science, and related disciplines
- **Semantic Scholar API**: Comprehensive paper metadata including citation counts, abstracts, and author information
- **LLM-Powered Keyword Extraction**: Dynamic extraction of domain-specific search terms for any research topic, enabling the system to work across diverse fields

### Multi-Agent Collaboration

The system employs seven specialized agents, each with distinct responsibilities:

| Agent | Primary Function |
|-------|------------------|
| Architect | Designs paper structure and section organization |
| Scholar | Writes publication-quality LaTeX content |
| Critic | Evaluates sections against IEEE standards |
| Artist | Generates TikZ diagrams and visualizations |
| Librarian | Manages BibTeX bibliography generation |
| Typesetter | Resolves LaTeX compilation errors |
| Integrator | Ensures document coherence and consistency |

### Quality Assurance Pipeline

Each section undergoes systematic review:

- Multi-dimensional scoring across technical depth, clarity, citations, formatting, and academic tone
- Iterative revision based on critic feedback with configurable thresholds
- Maximum revision attempts to prevent infinite loops
- Optional grammar checking via language-tool-python

### Thinking Mode Support

Full compatibility with reasoning-enabled models including Qwen3, DeepSeek-R1, and DeepSeek-V3:

- Real-time streaming of model reasoning processes
- Complete visibility into agent decision-making
- Separate logging of thinking traces for debugging and analysis

### Rich Terminal Interface

Professional terminal output using the Rich library:

- Colored panels with clear visual hierarchy
- Progress indicators for long-running operations
- Structured tables for configuration and results display
- Real-time streaming of model outputs during generation

---

## System Architecture

The generation pipeline follows a sequential flow with feedback loops for quality control:

```
                                    Input Topic
                                         |
                                         v
                              +---------------------+
                              |     Architect       |
                              | (Structure Design)  |
                              +---------------------+
                                         |
                                         v
                              +---------------------+
                              |  Keyword Extractor  |
                              | (Search Term Gen)   |
                              +---------------------+
                                         |
                                         v
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
          +------------------+                    +--------------------+
          |    arXiv API     |                    | Semantic Scholar   |
          +------------------+                    +--------------------+
                    |                                         |
                    +--------------------+--------------------+
                                         |
                                         v
                              +---------------------+
                              |      Scholar        |
                              | (Section Writing)   |
                              +---------------------+
                                         |
                                         v
                              +---------------------+
                              |       Critic        |
                              |   (Peer Review)     |
                              +---------------------+
                                         |
                              +----------+----------+
                              |                     |
                         [PASS]                 [REVISE]
                              |                     |
                              |                     v
                              |          +---------------------+
                              |          |      Scholar        |
                              |          |    (Revision)       |
                              |          +---------------------+
                              |                     |
                              +----------+----------+
                                         |
                                         v
                              +---------------------+
                              |       Artist        |
                              | (Diagram Creation)  |
                              +---------------------+
                                         |
                                         v
                              +---------------------+
                              |     Librarian       |
                              |  (Bibliography)     |
                              +---------------------+
                                         |
                                         v
                              +---------------------+
                              |     Typesetter      |
                              |  (Error Fixing)     |
                              +---------------------+
                                         |
                                         v
                              +---------------------+
                              |    PDF Compiler     |
                              +---------------------+
                                         |
                                         v
                                   Final Output
```

---

## Installation

### Prerequisites

Before installation, ensure the following are available on your system:

- Python 3.10 or higher
- Ollama installed and running
- A compatible LLM model pulled in Ollama
- LaTeX distribution (optional but recommended for PDF compilation)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Eeman1113/AgeSci.git
cd AgeSci
```

### Step 2: Install Python Dependencies

Install required packages using pip:

```bash
pip install ollama requests rich duckduckgo-search language-tool-python
```

Alternatively, use the requirements file if provided:

```bash
pip install -r requirements.txt
```

### Step 3: Install and Configure Ollama

Follow the installation instructions at [ollama.ai](https://ollama.ai) for your operating system.

After installation, pull a compatible model:

```bash
# Recommended for thinking mode support
ollama pull qwen3:latest

# Alternative with vision capabilities
ollama pull qwen3-vl:latest

# For enhanced reasoning tasks
ollama pull deepseek-r1:latest
```

### Step 4: Install LaTeX Distribution (Optional)

PDF compilation requires a LaTeX distribution. Installation varies by platform:

**macOS:**
```bash
brew install --cask mactex
```

**Ubuntu/Debian:**
```bash
sudo apt install texlive-full
```

**Fedora/RHEL:**
```bash
sudo dnf install texlive-scheme-full
```

**Windows:**

Download and install MiKTeX from [miktex.org](https://miktex.org) or TeX Live from [tug.org/texlive](https://tug.org/texlive).

---

## Dependencies

### Required Dependencies

| Package | Purpose | Installation |
|---------|---------|--------------|
| ollama | Local LLM interface | `pip install ollama` |
| requests | HTTP requests for APIs | `pip install requests` |
| rich | Terminal formatting | `pip install rich` |
| duckduckgo-search | Web search fallback | `pip install duckduckgo-search` |

### Optional Dependencies

| Package | Purpose | Installation |
|---------|---------|--------------|
| language-tool-python | Grammar checking | `pip install language-tool-python` |

Note: The grammar tool downloads approximately 1GB of language data on first use.

---

## Usage

### Basic Execution

Run the main script and provide your research topic when prompted:

```bash
python research.py
```

The system displays current configuration and prompts for input:

```
Enter your research topic: [Your detailed topic description]
```

### Effective Topic Descriptions

The system performs best with detailed topic descriptions that include:

- The specific problem or research question being addressed
- Relevant methods, techniques, or approaches to consider
- The target domain or application area
- Any constraints or specific angles to explore

**Example Input:**

```
Selective Concept Unlearning Without Catastrophic Forgetting: Most current 
methods like gradient ascent or ROME-style edits struggle to surgically 
remove specific knowledge without degrading general capabilities. A paper 
proposing a new technique that better preserves model utility while ensuring 
complete removal in LLMs.
```

### Monitoring Progress

During execution, the system displays:

- Current phase and agent activity
- Real-time model thinking (if enabled)
- Search results and citation counts
- Review scores and revision status
- Compilation progress and any errors

---

## Configuration

System behavior is controlled through the `Config` class at the top of the main script. Key parameters include:

### Model Settings

```python
model: str = "qwen3:latest"        # Ollama model identifier
temperature: float = 0.7           # Generation temperature (0.0-1.0)
enable_thinking: bool = True       # Enable thinking mode for compatible models
```

### Output Settings

```python
output_dir: str = "research_output"  # Directory for generated files
main_tex: str = "main.tex"           # Primary LaTeX filename
```

### Compilation Settings

```python
pdf_compiler: str = "pdflatex"    # LaTeX compiler command
compile_timeout: int = 60         # Maximum compilation time in seconds
```

### Quality Control Settings

```python
max_rewrites: int = 3             # Maximum revision attempts per section
min_citations_per_section: int = 3  # Minimum citations required
max_arxiv_results: int = 15       # Maximum papers to retrieve
```

### Robustness Settings

```python
max_retries: int = 3              # API retry attempts on failure
retry_delay: float = 1.0          # Delay between retries in seconds
enable_checkpointing: bool = True # Save intermediate progress
```

### Feature Toggles

```python
enable_grammar_check: bool = True   # Enable grammar validation
enable_latex_lint: bool = True      # Enable LaTeX syntax checking
verbose_logging: bool = True        # Enable detailed logging
log_thinking: bool = True           # Log model reasoning to file
```

### Model Compatibility Reference

| Model | Thinking Support | Recommended Use Case |
|-------|------------------|----------------------|
| qwen3:latest | Yes | General paper generation |
| qwen3-vl:latest | Yes | Topics requiring visual reasoning |
| deepseek-r1:latest | Yes | Complex technical topics |
| llama3:latest | No | Fast generation, simpler topics |
| mistral:latest | No | Balanced speed and quality |

---

## Output Structure

All generated files are saved to the configured output directory (default: `research_output/`):

```
research_output/
    main.tex            # Complete LaTeX document
    references.bib      # BibTeX bibliography file
    main.pdf            # Compiled PDF (if LaTeX available)
    research.log        # Progress and status log
    thinking.log        # Model reasoning traces
    verbose.log         # Complete input/output log
    plan.pkl            # Cached paper structure
    research.pkl        # Cached research results
    sections.pkl        # Cached section content
```

### File Descriptions

**main.tex**: The complete LaTeX document containing all sections, figures, tables, and formatting. Uses IEEEtran document class with standard academic packages.

**references.bib**: BibTeX entries for all cited papers. Includes complete metadata (title, authors, journal, year, URL) for each reference.

**main.pdf**: The compiled PDF document. Generated only if a LaTeX distribution is installed and compilation succeeds.

**Checkpoint Files (.pkl)**: Serialized Python objects containing intermediate results. Enable resumption of interrupted generation and caching of expensive operations.

**Log Files**: Detailed records of the generation process for debugging and analysis.

---

## Agent System

### Architect Agent

The Architect designs the paper structure based on the input topic:

- Generates a formal academic title
- Writes a 150-200 word abstract
- Identifies appropriate keywords
- Defines section organization with descriptions
- Marks sections requiring diagrams
- Estimates word counts for each section

Output format: JSON specification parsed by downstream agents.

### Scholar Agent

The primary content generation agent:

- Writes LaTeX-formatted section content
- Maintains formal academic tone throughout
- Incorporates provided citations appropriately
- Formats equations, tables, and itemized lists
- Follows IEEE style guidelines

The Scholar receives citation context and section descriptions from upstream agents.

### Critic Agent

Evaluates written content against quality criteria:

- **Technical Depth** (0-10): Substantive content and analysis
- **Clarity** (0-10): Organization and readability
- **Citations** (0-10): Appropriate use of references
- **Formatting** (0-10): Correct LaTeX syntax
- **Academic Tone** (0-10): Professional language

Sections scoring below 35/50 or flagged with major issues trigger revision cycles.

### Artist Agent

Generates TikZ diagrams for visual content:

- Architecture diagrams for methodology sections
- Flowcharts for process descriptions
- Data visualization frameworks
- Uses standard TikZ libraries for compatibility

### Librarian Agent

Manages bibliography generation:

- Formats BibTeX entries from paper metadata
- Validates required fields (title, author, year)
- Handles special characters and escaping
- Detects and manages duplicate entries

### Typesetter Agent

Resolves LaTeX compilation issues:

- Identifies syntax errors from compiler output
- Escapes special characters (%, &, $, #, _)
- Fixes malformed environments
- Repairs broken table and figure structures
- Preserves content while correcting markup

### Integrator Agent

Ensures document-wide coherence:

- Verifies consistent terminology usage
- Checks logical flow between sections
- Identifies redundant or contradictory content
- Suggests structural improvements

---

## Citation Pipeline

### Keyword Extraction

The system uses LLM-powered analysis to extract search terms:

1. The input topic is analyzed for technical terms and concepts
2. Domain-specific jargon and method names are identified
3. A balanced set of specific and general terms is generated
4. Terms are formatted for optimal database query performance

### Database Queries

Multiple search strategies maximize relevant results:

**Strategy 1**: Primary keywords (first 4 extracted terms)
**Strategy 2**: Alternative keyword combinations
**Strategy 3**: Title-derived terms (after structure generation)

### Deduplication

Results are deduplicated by normalized title to prevent redundant citations.

### Citation Key Generation

Unique keys follow the format: `{firstAuthorLastName}{year}{firstTitleWord}`

Example: `gupta2024model` for a 2024 paper by Gupta titled "Model Editing..."

### Context Injection

Available citations are provided to the Scholar agent as structured context:

```
AVAILABLE CITATIONS (use these exact keys):
- \cite{gupta2024model}: Model Editing at Scale leads to... (2024)
- \cite{doan2020a}: A Theoretical Analysis of Catastrophic... (2020)
```

### Unknown Citation Handling

If a section references an unfound citation key, the Librarian generates a placeholder:

```bibtex
@misc{unknown2024citation,
    title = {[Citation needed]},
    author = {Unknown},
    year = {2024},
    note = {Citation key referenced but source not found}
}
```

This ensures compilation succeeds while flagging citations requiring manual verification.

---

## Quality Control

### Review Scoring

Each section is evaluated across five dimensions:

| Dimension | Weight | Criteria |
|-----------|--------|----------|
| Technical Depth | 10 | Substantive analysis, appropriate complexity |
| Clarity | 10 | Logical organization, readable prose |
| Citations | 10 | Relevant references, proper integration |
| Formatting | 10 | Valid LaTeX, consistent style |
| Academic Tone | 10 | Formal language, objective presentation |

### Pass/Fail Determination

- **Pass**: Total score >= 35 AND no major issues flagged
- **Revise**: Total score < 35 OR major issues present

### Revision Process

Failed sections enter a revision cycle:

1. Critic feedback is compiled into specific improvement points
2. Scholar rewrites the section addressing each point
3. Revised content is re-evaluated
4. Process repeats until pass or maximum attempts reached

### Grammar Checking

When enabled, sections are analyzed for:

- Spelling errors
- Grammar violations
- Style inconsistencies
- Punctuation issues

Issues are logged and optionally incorporated into revision feedback.

---

## Example Output

The repository includes a complete example output in the `research_output/` directory:

**Topic**: Selective Concept Unlearning Without Catastrophic Forgetting in LLMs

**Generated Paper Structure**:
- Title: "Selective Concept Unlearning in Large Language Models: A Dual-Path Architecture to Prevent Catastrophic Forgetting"
- Sections: Introduction, Related Work, Methodology, Results and Discussion, Conclusion
- Citations: 18 real papers from arXiv (2018-2025)
- Figures: TikZ architecture diagram
- Tables: Quantitative comparison of methods

**Citation Examples from Generated Paper**:
```bibtex
@article{gupta2024model,
    title={Model Editing at Scale leads to Gradual and Catastrophic Forgetting},
    author={Akshat Gupta and Anurag Rao and Gopala Anumanchipalli},
    journal={arXiv preprint arXiv:2401.07453v4},
    year={2024},
    url={http://arxiv.org/abs/2401.07453v4}
}

@article{doan2020a,
    title={A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix},
    author={Thang Doan and Mehdi Bennani and Bogdan Mazoure and others},
    journal={arXiv preprint arXiv:2010.04003v2},
    year={2020},
    url={http://arxiv.org/abs/2010.04003v2}
}
```

---

## Limitations and Considerations

### Technical Limitations

1. **Citation Relevance**: While all citations reference real papers, the system cannot guarantee deep semantic relevance to specific claims. Citations are selected based on keyword matching and may require manual verification.

2. **Factual Accuracy**: Generated content synthesizes patterns from training data and retrieved abstracts. Claims should be verified against primary sources before use.

3. **Novel Contributions**: The system produces literature synthesis and structured writing but does not conduct original research or generate genuinely novel findings.

4. **Domain Coverage**: Performance varies by field based on arXiv coverage. Fields with limited preprint culture may yield fewer relevant citations.

5. **Compilation Dependencies**: PDF generation requires local LaTeX installation. Complex diagrams may require manual adjustment.

6. **Model Limitations**: Output quality depends on the underlying LLM capabilities. Smaller models may produce less coherent or technically accurate content.

### Ethical Considerations

This tool is designed and intended for:

- Research exploration and ideation
- Understanding academic paper structure and conventions
- Generating initial drafts for human refinement
- Educational purposes and writing assistance
- Rapid prototyping of research directions

This tool should not be used to:

- Submit generated papers as original work without disclosure
- Circumvent academic integrity requirements
- Generate papers for fraudulent publication
- Misrepresent AI-generated content as human-authored

All generated content should be thoroughly reviewed, verified against primary sources, and substantially revised before any academic submission. Users bear responsibility for ensuring compliance with relevant academic integrity policies.

---

## Troubleshooting

### Common Issues and Solutions

**Issue: Model not responding or connection refused**

```
Solution: Ensure Ollama is running
$ ollama serve

Verify model availability:
$ ollama list
```

**Issue: No papers found for topic**

```
Possible causes:
- Topic too specific or uses non-standard terminology
- arXiv coverage limited for the domain
- Network connectivity issues

Solutions:
- Use broader, more standard academic terminology
- Check arXiv directly for relevant papers
- Verify network access to api.arxiv.org
```

**Issue: LaTeX compilation fails**

```
Common causes:
- Unescaped special characters in generated content
- Missing LaTeX packages
- Malformed TikZ diagrams

Solutions:
- Review error messages in compilation output
- Check main.tex for obvious syntax errors
- Install missing packages via your TeX distribution
- Disable diagram generation if issues persist
```

**Issue: Grammar tool initialization fails**

```
Solution: The tool downloads ~1GB on first use
$ pip install language-tool-python --upgrade

Alternative: Disable grammar checking
enable_grammar_check: bool = False
```

**Issue: Out of memory during generation**

```
Solutions:
- Use a smaller model (e.g., 7B instead of 70B)
- Reduce max_arxiv_results in configuration
- Close other memory-intensive applications
- Consider using quantized model variants
```

**Issue: Generation stuck or extremely slow**

```
Possible causes:
- Model too large for available hardware
- Network timeouts during API calls
- Infinite revision loops

Solutions:
- Monitor system resources during generation
- Check network connectivity
- Reduce max_rewrites to limit revision cycles
- Use a faster/smaller model
```

---

## Contributing

Contributions to AgeSci are welcome. Areas particularly suited for improvement include:

### Feature Additions
- Additional academic database integrations (IEEE Xplore, PubMed, Google Scholar, ACL Anthology)
- Support for additional document formats (ACM, Springer, Nature, AAAI)
- Enhanced diagram generation with more complex visualizations
- Web-based interface for non-technical users
- Batch processing for multiple topics

### Quality Improvements
- Improved citation relevance scoring using semantic similarity
- Better handling of multi-concept and interdisciplinary topics
- Enhanced grammar and style checking
- More sophisticated revision feedback

### Technical Enhancements
- Parallel agent execution for faster generation
- Distributed processing support
- Model-agnostic backend supporting multiple LLM providers
- Improved caching and checkpointing

### Documentation
- Additional usage examples
- Video tutorials
- Troubleshooting guides for specific platforms

Please submit issues for bug reports and feature requests. Pull requests should include appropriate tests and documentation updates.

---

## License

This project is provided as-is for research and educational purposes. Users must ensure compliance with:

- Terms of service for arXiv API
- Terms of service for Semantic Scholar API
- Academic integrity policies of their institutions
- Applicable laws regarding AI-generated content

---

## Citation

If you use AgeSci in your research or projects, please consider citing:

```bibtex
@software{agesci2025,
    title = {AgeSci: Autonomous Multi-Agent Research Paper Generation System},
    author = {Majumder, Eeman},
    year = {2025},
    url = {https://github.com/Eeman1113/AgeSci},
    note = {Autonomous system for generating IEEE-format academic papers using multi-agent LLM coordination}
}
```

---

## Acknowledgments

AgeSci builds upon and integrates several open-source projects and public APIs:

- [Ollama](https://ollama.ai) for local LLM inference
- [arXiv API](https://arxiv.org/help/api) for academic paper access
- [Semantic Scholar API](https://api.semanticscholar.org) for citation metadata
- [Rich](https://github.com/Textualize/rich) for terminal formatting
- [language-tool-python](https://github.com/jxmorris12/language_tool_python) for grammar checking

---

For questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/Eeman1113/AgeSci).
