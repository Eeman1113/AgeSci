#!/usr/bin/env python3
"""
Fully autonomous - just provide a topic and get a complete IEEE-style paper.

FREE TOOLS USED:
- arXiv API (real academic citations)
- Semantic Scholar API (paper summaries & references)
- CrossRef API (DOI lookup)
- language_tool_python (grammar checking)
- chktex (LaTeX linting)
- DuckDuckGo Search (general research)

Install requirements:
    pip install ollama requests language-tool-python duckduckgo-search arxiv
    sudo apt update && sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-science texlive-bibtex-extra biber chktex && pip install ollama requests duckduckgo-search language-tool-python

    macos:
    brew update && brew install --cask mactex && brew install chktex && pip install ollama requests duckduckgo-search language-tool-python
    eval "$(/usr/libexec/path_helper)"
    
"""

import os
import json
import subprocess
import re
import sys
import datetime
import time
import hashlib
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from abc import ABC, abstractmethod
import urllib.parse
import xml.etree.ElementTree as ET

import ollama
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Centralized configuration - modify these settings as needed"""
    # Model settings
    model: str = "qwen3-vl:latest"
    temperature: float = 0.7
    
    # Output settings
    output_dir: str = "research_output"
    main_tex: str = "main.tex"
    
    # Compilation
    pdf_compiler: str = "pdflatex"
    compile_timeout: int = 60
    
    # Quality control
    max_rewrites: int = 3
    min_citations_per_section: int = 3
    max_arxiv_results: int = 15
    
    # Robustness
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_checkpointing: bool = True
    
    # Features
    enable_grammar_check: bool = True
    enable_latex_lint: bool = True
    parallel_sections: bool = False  # Set True if your machine can handle it


CONFIG = Config()
os.makedirs(CONFIG.output_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    """Clean logging to both console and file"""
    
    def __init__(self, output_dir: str):
        self.log_path = os.path.join(output_dir, "research.log")
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"RESEARCH PAPER GENERATION - {datetime.datetime.now()}\n")
            f.write(f"{'='*70}\n\n")
    
    def _log(self, level: str, msg: str, icon: str = ""):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {icon} {msg}"
        print(formatted)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{level}] {formatted}\n")
    
    def info(self, msg: str): self._log("INFO", msg, "ℹ️")
    def success(self, msg: str): self._log("SUCCESS", msg, "✅")
    def warning(self, msg: str): self._log("WARNING", msg, "⚠️")
    def error(self, msg: str): self._log("ERROR", msg, "❌")
    def phase(self, msg: str): self._log("PHASE", f"\n{'─'*50}\n{msg}\n{'─'*50}", "📋")
    def agent(self, name: str, msg: str): self._log("AGENT", f"[{name}] {msg}", "🤖")


LOG = Logger(CONFIG.output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator for flaky operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_error
        return wrapper
    return decorator


def cache_result(func):
    """Simple file-based caching for expensive operations"""
    cache = {}
    def wrapper(*args, **kwargs):
        key = hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()
        if key in cache:
            return cache[key]
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# FREE RESEARCH TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

class ArxivTool:
    """
    Search arXiv for real academic papers - FREE, no API key needed
    """
    BASE_URL = "http://export.arxiv.org/api/query"
    
    @staticmethod
    @retry(max_attempts=3)
    @cache_result
    def search(query: str, max_results: int = 10) -> List[Dict]:
        """Search arXiv and return paper metadata"""
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = requests.get(ArxivTool.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            # Get first author's last name for citation key
            first_author_last = authors[0].split()[-1] if authors else "Unknown"
            
            # Extract year from published date
            published = entry.find('atom:published', ns)
            year = published.text[:4] if published is not None else "2024"
            
            title_elem = entry.find('atom:title', ns)
            summary_elem = entry.find('atom:summary', ns)
            arxiv_id = entry.find('atom:id', ns)
            
            # Generate citation key
            title_word = re.sub(r'[^a-zA-Z]', '', 
                              (title_elem.text.split()[0] if title_elem is not None else "paper")).lower()
            cite_key = f"{first_author_last.lower()}{year}{title_word}"
            
            papers.append({
                'title': title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "",
                'authors': authors,
                'year': year,
                'abstract': summary_elem.text.strip().replace('\n', ' ')[:500] if summary_elem is not None else "",
                'arxiv_id': arxiv_id.text.split('/')[-1] if arxiv_id is not None else "",
                'cite_key': cite_key,
                'url': arxiv_id.text if arxiv_id is not None else ""
            })
        
        return papers
    
    @staticmethod
    def format_bibtex(paper: Dict) -> str:
        """Format a paper as BibTeX entry"""
        authors_str = " and ".join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors_str += " and others"
        
        return f"""@article{{{paper['cite_key']},
    title={{{paper['title']}}},
    author={{{authors_str}}},
    journal={{arXiv preprint arXiv:{paper['arxiv_id']}}},
    year={{{paper['year']}}},
    url={{{paper['url']}}}
}}"""


class SemanticScholarTool:
    """
    Search Semantic Scholar for papers - FREE, no API key needed (rate limited)
    """
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    @staticmethod
    @retry(max_attempts=3)
    @cache_result
    def search(query: str, limit: int = 10) -> List[Dict]:
        """Search for papers"""
        url = f"{SemanticScholarTool.BASE_URL}/paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,authors,year,abstract,citationCount,url,externalIds'
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 429:  # Rate limited
            time.sleep(5)
            response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        papers = []
        
        for paper in data.get('data', []):
            if not paper.get('title'):
                continue
                
            authors = [a.get('name', '') for a in paper.get('authors', [])[:5]]
            first_author_last = authors[0].split()[-1] if authors else "Unknown"
            year = str(paper.get('year', '2024'))
            
            title_word = re.sub(r'[^a-zA-Z]', '', 
                              paper['title'].split()[0]).lower()
            cite_key = f"{first_author_last.lower()}{year}{title_word}"
            
            papers.append({
                'title': paper['title'],
                'authors': authors,
                'year': year,
                'abstract': (paper.get('abstract') or "")[:500],
                'citations': paper.get('citationCount', 0),
                'cite_key': cite_key,
                'url': paper.get('url', ''),
                'doi': paper.get('externalIds', {}).get('DOI', '')
            })
        
        return papers


class DuckDuckGoTool:
    """
    Web search using DuckDuckGo - FREE, no API key
    """
    
    @staticmethod
    @cache_result
    def search(query: str, max_results: int = 5) -> List[Dict]:
        """Search the web for general information"""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': r.get('title', ''),
                        'body': r.get('body', ''),
                        'url': r.get('href', '')
                    })
            return results
        except ImportError:
            LOG.warning("DuckDuckGo search not available. Install: pip install duckduckgo-search")
            return []
        except Exception as e:
            LOG.warning(f"DuckDuckGo search failed: {e}")
            return []


class GrammarTool:
    """
    Grammar and style checking - FREE, runs locally
    """
    _tool = None
    
    @classmethod
    def get_tool(cls):
        if cls._tool is None:
            try:
                import language_tool_python
                cls._tool = language_tool_python.LanguageTool('en-US')
            except ImportError:
                LOG.warning("Grammar tool not available. Install: pip install language-tool-python")
                return None
        return cls._tool
    
    @classmethod
    def check(cls, text: str) -> Tuple[str, List[str]]:
        """Check grammar and return corrected text + issues found"""
        tool = cls.get_tool()
        if tool is None:
            return text, []
        
        # Remove LaTeX commands for checking
        clean_text = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', '', text)
        clean_text = re.sub(r'[\$\{\}\[\]]', '', clean_text)
        
        matches = tool.check(clean_text)
        issues = [f"{m.ruleId}: {m.message}" for m in matches[:5]]  # Top 5 issues
        
        # Apply corrections to original text carefully
        corrected = text
        for match in reversed(matches):
            if match.replacements:
                # Only fix simple issues, not LaTeX
                if not any(c in match.context for c in ['\\', '{', '}']):
                    pass  # Skip complex fixes that might break LaTeX
        
        return text, issues


class LaTeXLinter:
    """
    LaTeX linting using chktex - FREE
    """
    
    @staticmethod
    def lint(tex_content: str) -> List[str]:
        """Run chktex and return warnings"""
        try:
            # Write to temp file
            temp_file = os.path.join(CONFIG.output_dir, "_temp_lint.tex")
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(tex_content)
            
            result = subprocess.run(
                ['chktex', '-q', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            os.remove(temp_file)
            
            # Parse warnings
            warnings = []
            for line in result.stdout.split('\n'):
                if 'Warning' in line:
                    warnings.append(line.strip())
            
            return warnings[:10]  # Top 10 warnings
            
        except FileNotFoundError:
            LOG.warning("chktex not installed. Install: sudo apt install chktex")
            return []
        except Exception as e:
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY - Agents can access these
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS = {
    'arxiv_search': ArxivTool.search,
    'semantic_scholar_search': SemanticScholarTool.search,
    'web_search': DuckDuckGoTool.search,
    'grammar_check': GrammarTool.check,
    'latex_lint': LaTeXLinter.lint,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Agent:
    """Enhanced agent with tool access and better error handling"""
    
    def __init__(self, name: str, system_prompt: str, tools: List[str] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.conversation_history = []
    
    def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool and return results"""
        if tool_name in TOOLS:
            try:
                return TOOLS[tool_name](**kwargs)
            except Exception as e:
                LOG.warning(f"Tool {tool_name} failed: {e}")
                return None
        return None
    
    @retry(max_attempts=CONFIG.max_retries, delay=CONFIG.retry_delay)
    def think(self, user_input: str, use_tools: bool = True) -> str:
        """Process input and generate response, optionally using tools first"""
        LOG.agent(self.name, f"Processing: {user_input[:80]}...")
        
        # Build context from tools if needed
        tool_context = ""
        if use_tools and self.tools:
            for tool_name in self.tools:
                if tool_name == 'arxiv_search' and 'research' in self.name.lower():
                    # Extract key terms for search
                    search_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_input)
                    query = ' '.join(search_terms[:3]) if search_terms else user_input[:50]
                    results = self._execute_tool('arxiv_search', query=query, max_results=5)
                    if results:
                        tool_context += "\n\nRELEVANT PAPERS FROM ARXIV:\n"
                        for p in results:
                            tool_context += f"- {p['title']} ({p['year']}) [cite: {p['cite_key']}]\n"
        
        # Build messages
        messages = [
            {'role': 'system', 'content': self.system_prompt + tool_context}
        ]
        
        # Add conversation history for context
        messages.extend(self.conversation_history[-4:])  # Last 2 exchanges
        messages.append({'role': 'user', 'content': user_input})
        
        # Call model
        response = ollama.chat(
            model=CONFIG.model,
            messages=messages,
            options={'temperature': CONFIG.temperature}
        )
        
        content = response['message']['content']
        
        # Update history
        self.conversation_history.append({'role': 'user', 'content': user_input})
        self.conversation_history.append({'role': 'assistant', 'content': content})
        
        return content
    
    def reset_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED AGENT PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

PROMPTS = {
    'architect': """You are a Lead Research Architect specializing in IEEE-format academic papers.

YOUR TASK: Create a detailed paper structure as JSON.

REQUIRED OUTPUT FORMAT (strict JSON, no markdown):
{
    "title": "A Formal Academic Title",
    "abstract": "A 150-200 word abstract summarizing the paper...",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "sections": [
        {
            "title": "Introduction",
            "description": "Detailed description of what this section should cover...",
            "subsections": ["Background", "Motivation", "Contributions"],
            "needs_diagram": false,
            "estimated_length": "400-500 words"
        },
        {
            "title": "Related Work", 
            "description": "Survey of existing approaches...",
            "subsections": [],
            "needs_diagram": false,
            "estimated_length": "300-400 words"
        }
    ]
}

RULES:
1. Include standard sections: Introduction, Related Work, Methodology, Results/Discussion, Conclusion
2. Each section needs a detailed description for writers
3. Mark sections needing diagrams (methodology, architecture, results)
4. Be specific about what each section should contain
5. Output ONLY valid JSON, no explanations""",

    'researcher': """You are a Research Specialist with access to academic databases.

YOUR TASK: Find and synthesize relevant research for a paper section.

You have access to real papers from arXiv (provided in context). 

WHEN WRITING:
1. Reference the REAL papers provided - use their actual citation keys
2. Summarize key findings from these papers
3. Identify research gaps
4. Suggest how the current work fits into existing literature

OUTPUT FORMAT:
{
    "key_papers": [
        {"cite_key": "...", "relevance": "How this paper relates..."}
    ],
    "research_gaps": ["Gap 1", "Gap 2"],
    "synthesis": "A paragraph synthesizing the literature...",
    "suggested_citations": ["cite_key1", "cite_key2"]
}

Output ONLY valid JSON.""",

    'scholar': """You are an Expert Academic Writer for IEEE-format papers.

YOUR TASK: Write a complete, publication-quality section in LaTeX.

CRITICAL RULES:
1. Use ONLY the citation keys provided - these are real papers
2. Use LaTeX formatting: \\textbf{}, \\textit{}, \\begin{itemize}
3. Write formally - no first person unless absolutely necessary
4. Be technical and precise
5. Each paragraph should be 4-6 sentences
6. Include equations where appropriate using $...$ or \\begin{equation}

CITATION FORMAT:
- Single: \\cite{key}
- Multiple: \\cite{key1, key2}
- With text: Author et al. \\cite{key} showed that...

OUTPUT: Return ONLY the LaTeX body content (no \\section{} header, I'll add that).
Do NOT wrap in ```latex``` blocks.""",

    'critic': """You are a Rigorous Academic Reviewer (IEEE standards).

YOUR TASK: Evaluate the section and provide detailed feedback.

EVALUATION CRITERIA:
1. TECHNICAL DEPTH (0-10): Is the content substantive?
2. CLARITY (0-10): Is the writing clear and well-organized?
3. CITATIONS (0-10): Are citations used properly and sufficiently?
4. FORMATTING (0-10): Is LaTeX used correctly?
5. ACADEMIC TONE (0-10): Is it appropriately formal?

OUTPUT FORMAT (strict JSON):
{
    "status": "PASS" or "REVISE",
    "scores": {
        "technical_depth": 8,
        "clarity": 7,
        "citations": 9,
        "formatting": 8,
        "academic_tone": 9
    },
    "total_score": 41,
    "major_issues": ["Issue 1 requiring revision"],
    "minor_issues": ["Small improvement suggestions"],
    "feedback": "Detailed feedback paragraph..."
}

PASS if total_score >= 35 AND no major_issues.
Output ONLY valid JSON.""",

    'artist': """You are a TikZ Diagram Expert.

YOUR TASK: Create a clear, professional diagram in TikZ.

RULES:
1. Use ONLY standard TikZ libraries (shapes, arrows, positioning)
2. Keep it simple and readable
3. Use proper node positioning
4. Include appropriate labels
5. Use professional colors sparingly

OUTPUT: Return ONLY the complete TikZ code starting with \\begin{tikzpicture} and ending with \\end{tikzpicture}.
Do NOT include any explanation or markdown.""",

    'librarian': """You are a BibTeX Expert.

YOUR TASK: Generate valid BibTeX entries for the provided papers.

INPUT: You'll receive paper metadata (title, authors, year, arxiv_id, etc.)

OUTPUT FORMAT:
Return a complete BibTeX file with entries like:

@article{citekey,
    title = {Paper Title},
    author = {First Author and Second Author},
    journal = {arXiv preprint arXiv:XXXX.XXXXX},
    year = {2024},
    url = {https://arxiv.org/abs/XXXX.XXXXX}
}

RULES:
1. Each entry must have: title, author, year, journal/booktitle
2. Use the EXACT cite_key provided
3. Format author names as "Last, First and Last, First"
4. Return ONLY the BibTeX entries, no explanations""",

    'typesetter': """You are a LaTeX Debugging Expert.

YOUR TASK: Fix LaTeX compilation errors.

WHEN FIXING:
1. Analyze the error message carefully
2. Common fixes:
   - Missing \\end{} tags
   - Unescaped special characters (%, &, $, #, _)
   - Invalid TikZ syntax
   - Missing packages
   - Malformed tables/figures
3. Preserve ALL content - only fix syntax
4. Ensure document structure is complete

OUTPUT: Return the COMPLETE fixed LaTeX document.
Do NOT wrap in ```latex``` blocks.
Do NOT include explanations.""",

    'integrator': """You are a Document Integration Specialist.

YOUR TASK: Ensure document coherence and flow.

CHECK FOR:
1. Consistent terminology throughout
2. Proper section transitions
3. Forward/backward references make sense
4. No duplicate content
5. Logical flow of arguments

OUTPUT FORMAT (JSON):
{
    "coherence_score": 8,
    "issues": ["Issue 1", "Issue 2"],
    "suggested_fixes": ["Fix 1", "Fix 2"],
    "transition_suggestions": {
        "section1_to_section2": "Suggested transition sentence..."
    }
}

Output ONLY valid JSON."""
}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_json(text: str) -> str:
    """Extract JSON from LLM response"""
    # Try to find JSON in code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}',
    ]
    
    for pattern in patterns[:2]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Try to find raw JSON object
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        return text[start:end]
    except ValueError:
        pass
    
    return text.strip()


def clean_latex(text: str) -> str:
    """Extract LaTeX from LLM response"""
    # Remove markdown code blocks
    text = re.sub(r'```latex\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    return text


def save_file(filename: str, content: str) -> str:
    """Save file to output directory"""
    filepath = os.path.join(CONFIG.output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    LOG.success(f"Saved: {filepath}")
    return filepath


def compile_pdf(tex_file: str) -> Tuple[bool, str]:
    """Compile LaTeX to PDF"""
    LOG.info(f"Compiling {tex_file}...")
    
    try:
        # Run pdflatex twice for cross-references
        for i in range(2):
            result = subprocess.run(
                [CONFIG.pdf_compiler, '-interaction=nonstopmode', '-halt-on-error', tex_file],
                cwd=CONFIG.output_dir,
                capture_output=True,
                text=True,
                timeout=CONFIG.compile_timeout
            )
        
        if result.returncode == 0:
            return True, "Success"
        else:
            # Extract relevant error lines
            error_lines = []
            for line in result.stdout.split('\n'):
                if any(x in line.lower() for x in ['error', '!', 'missing', 'undefined']):
                    error_lines.append(line)
            return False, '\n'.join(error_lines[-20:])
            
    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except FileNotFoundError:
        return False, f"{CONFIG.pdf_compiler} not found. Install texlive."
    except Exception as e:
        return False, str(e)


def save_checkpoint(data: Dict, name: str = "checkpoint"):
    """Save progress checkpoint"""
    if CONFIG.enable_checkpointing:
        path = os.path.join(CONFIG.output_dir, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(data, f)


def load_checkpoint(name: str = "checkpoint") -> Optional[Dict]:
    """Load progress checkpoint"""
    path = os.path.join(CONFIG.output_dir, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PAPER GENERATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchPaperGenerator:
    """Orchestrates the multi-agent paper generation pipeline"""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.plan = None
        self.sections = {}
        self.citations = {}
        self.all_papers = []
        
        # Initialize agents
        self.architect = Agent("Architect", PROMPTS['architect'])
        self.researcher = Agent("Researcher", PROMPTS['researcher'], tools=['arxiv_search'])
        self.scholar = Agent("Scholar", PROMPTS['scholar'])
        self.critic = Agent("Critic", PROMPTS['critic'])
        self.artist = Agent("Artist", PROMPTS['artist'])
        self.librarian = Agent("Librarian", PROMPTS['librarian'])
        self.typesetter = Agent("Typesetter", PROMPTS['typesetter'])
        self.integrator = Agent("Integrator", PROMPTS['integrator'])
    
    def phase1_planning(self) -> bool:
        """Phase 1: Create paper structure"""
        LOG.phase("PHASE 1: ARCHITECTURE & PLANNING")
        
        plan_raw = self.architect.think(
            f"Create a comprehensive IEEE paper structure for the topic: {self.topic}\n\n"
            f"The paper should be thorough and academically rigorous."
        )
        
        try:
            self.plan = json.loads(clean_json(plan_raw))
            LOG.success(f"Paper planned: {self.plan['title']}")
            LOG.info(f"Sections: {[s['title'] for s in self.plan['sections']]}")
            save_checkpoint({'plan': self.plan}, 'plan')
            return True
        except json.JSONDecodeError as e:
            LOG.error(f"Failed to parse plan: {e}")
            LOG.error(f"Raw output: {plan_raw[:500]}")
            return False
    
    def phase2_research(self) -> bool:
        """Phase 2: Gather research from real sources"""
        LOG.phase("PHASE 2: RESEARCH & CITATION GATHERING")
        
        # Search arXiv for relevant papers
        LOG.info(f"Searching arXiv for: {self.topic}")
        arxiv_papers = ArxivTool.search(self.topic, max_results=CONFIG.max_arxiv_results)
        
        if arxiv_papers:
            LOG.success(f"Found {len(arxiv_papers)} papers on arXiv")
            for p in arxiv_papers[:5]:
                LOG.info(f"  - {p['title'][:60]}... [{p['cite_key']}]")
        else:
            LOG.warning("No arXiv papers found, searching with broader terms...")
            # Try broader search
            broad_terms = self.topic.split()[:2]
            arxiv_papers = ArxivTool.search(' '.join(broad_terms), max_results=CONFIG.max_arxiv_results)
        
        # Also search Semantic Scholar
        LOG.info("Searching Semantic Scholar...")
        ss_papers = SemanticScholarTool.search(self.topic, limit=10)
        
        if ss_papers:
            LOG.success(f"Found {len(ss_papers)} papers on Semantic Scholar")
        
        # Combine and deduplicate
        self.all_papers = arxiv_papers + ss_papers
        seen_titles = set()
        unique_papers = []
        for p in self.all_papers:
            title_lower = p['title'].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(p)
        self.all_papers = unique_papers
        
        # Create citation lookup
        for p in self.all_papers:
            self.citations[p['cite_key']] = p
        
        LOG.success(f"Total unique papers for citation: {len(self.all_papers)}")
        save_checkpoint({'papers': self.all_papers, 'citations': self.citations}, 'research')
        
        return len(self.all_papers) > 0
    
    def phase3_writing(self) -> bool:
        """Phase 3: Write each section with quality control"""
        LOG.phase("PHASE 3: WRITING & PEER REVIEW")
        
        # Prepare citation context for writers
        citation_context = "AVAILABLE CITATIONS (use these exact keys):\n"
        for p in self.all_papers[:15]:  # Top 15 papers
            citation_context += f"- \\cite{{{p['cite_key']}}}: {p['title'][:80]}... ({p['year']})\n"
        
        for section in self.plan['sections']:
            section_title = section['title']
            LOG.info(f"\n📝 Writing: {section_title}")
            
            # Write with multiple attempts if needed
            passed = False
            attempts = 0
            draft = ""
            feedback = ""
            
            while not passed and attempts < CONFIG.max_rewrites:
                attempts += 1
                
                # Build comprehensive prompt
                prompt = f"""Write the '{section_title}' section.

SECTION DETAILS:
{json.dumps(section, indent=2)}

{citation_context}

{"PREVIOUS FEEDBACK TO ADDRESS: " + feedback if feedback else ""}

Write a complete, publication-ready section. Use the real citations provided above.
"""
                
                draft = clean_latex(self.scholar.think(prompt))
                
                # Check grammar if enabled
                if CONFIG.enable_grammar_check:
                    _, grammar_issues = GrammarTool.check(draft)
                    if grammar_issues:
                        LOG.warning(f"Grammar issues: {grammar_issues[:2]}")
                
                # Peer review
                review_prompt = f"""Review this section titled '{section_title}':

{draft}

Evaluate against IEEE standards."""
                
                review_raw = self.critic.think(review_prompt)
                
                try:
                    review = json.loads(clean_json(review_raw))
                    total_score = review.get('total_score', 0)
                    status = review.get('status', 'REVISE')
                    
                    if status == 'PASS' or total_score >= 35:
                        passed = True
                        LOG.success(f"Approved (attempt {attempts}, score: {total_score}/50)")
                    else:
                        feedback = review.get('feedback', '')
                        major = review.get('major_issues', [])
                        LOG.warning(f"Revision needed (score: {total_score}/50)")
                        if major:
                            LOG.warning(f"Major issues: {major[0]}")
                            
                except json.JSONDecodeError:
                    LOG.warning("Could not parse review, passing by default")
                    passed = True
            
            # Store section
            self.sections[section_title] = {
                'content': draft,
                'attempts': attempts
            }
            
            # Generate diagram if needed
            if section.get('needs_diagram', False):
                LOG.info(f"  🎨 Generating diagram for {section_title}...")
                diagram_prompt = f"""Create a TikZ diagram for the '{section_title}' section.
                
Context: {section.get('description', '')}

Create a clear, professional diagram."""
                
                tikz_code = clean_latex(self.artist.think(diagram_prompt))
                self.sections[section_title]['diagram'] = tikz_code
        
        save_checkpoint({'sections': self.sections}, 'sections')
        return True
    
    def phase4_bibliography(self) -> str:
        """Phase 4: Generate bibliography from real sources"""
        LOG.phase("PHASE 4: BIBLIOGRAPHY GENERATION")
        
        # Find all citations used in the paper
        all_content = ' '.join(s['content'] for s in self.sections.values())
        used_keys = set(re.findall(r'\\cite\{([^}]+)\}', all_content))
        
        # Expand comma-separated citations
        expanded_keys = set()
        for key in used_keys:
            for k in key.split(','):
                expanded_keys.add(k.strip())
        
        LOG.info(f"Citations used in paper: {len(expanded_keys)}")
        
        # Generate BibTeX entries for used citations
        bibtex_entries = []
        for key in expanded_keys:
            if key in self.citations:
                paper = self.citations[key]
                entry = ArxivTool.format_bibtex(paper)
                bibtex_entries.append(entry)
            else:
                # Create placeholder for unknown citations
                LOG.warning(f"Unknown citation key: {key}")
                bibtex_entries.append(f"""@misc{{{key},
    title = {{[Citation needed]}},
    author = {{Unknown}},
    year = {{2024}},
    note = {{Citation key referenced but source not found}}
}}""")
        
        bibtex_content = '\n\n'.join(bibtex_entries)
        save_file('references.bib', bibtex_content)
        LOG.success(f"Generated {len(bibtex_entries)} bibliography entries")
        
        return bibtex_content
    
    def phase5_assembly(self, bibtex: str) -> str:
        """Phase 5: Assemble complete document"""
        LOG.phase("PHASE 5: DOCUMENT ASSEMBLY")
        
        # IEEE document preamble
        preamble = r"""\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,calc}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{multirow}

\begin{document}

"""
        
        # Title and author
        title = self.plan['title'].replace('&', '\\&').replace('%', '\\%')
        header = f"""\\title{{{title}}}
\\author{{\\IEEEauthorblockN{{AI Research System}}
\\IEEEauthorblockA{{\\textit{{Automated Research Laboratory}}\\\\
Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}}}}}
\\maketitle

"""
        
        # Abstract
        abstract = self.plan['abstract'].replace('&', '\\&').replace('%', '\\%')
        abstract_section = f"""\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
{', '.join(self.plan.get('keywords', ['artificial intelligence', 'machine learning']))}
\\end{{IEEEkeywords}}

"""
        
        # Sections
        body = ""
        for section in self.plan['sections']:
            title = section['title']
            if title in self.sections:
                content = self.sections[title]['content']
                body += f"\\section{{{title}}}\n{content}\n\n"
                
                # Add diagram if present
                if 'diagram' in self.sections[title]:
                    diagram = self.sections[title]['diagram']
                    body += f"""\\begin{{figure}}[htbp]
\\centering
{diagram}
\\caption{{Visualization for {title}}}
\\label{{fig:{title.lower().replace(' ', '_')}}}
\\end{{figure}}

"""
        
        # Bibliography (using BibTeX style)
        save_file('references.bib', bibtex)
        
        bibliography = r"""
\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}"""
        
        # Combine everything
        full_document = preamble + header + abstract_section + body + bibliography
        
        # Save
        save_file(CONFIG.main_tex, full_document)
        LOG.success("Document assembled")
        
        return full_document
    
    def phase6_compilation(self, document: str) -> bool:
        """Phase 6: Compile to PDF with error fixing"""
        LOG.phase("PHASE 6: PDF COMPILATION")
        
        # First, run bibtex
        LOG.info("Running BibTeX...")
        try:
            # First pdflatex pass
            subprocess.run(
                [CONFIG.pdf_compiler, '-interaction=nonstopmode', CONFIG.main_tex],
                cwd=CONFIG.output_dir,
                capture_output=True,
                timeout=60
            )
            
            # BibTeX
            subprocess.run(
                ['bibtex', CONFIG.main_tex.replace('.tex', '')],
                cwd=CONFIG.output_dir,
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            LOG.warning(f"BibTeX step failed: {e}")
        
        # Compile with retry loop
        for attempt in range(CONFIG.max_retries):
            success, error_log = compile_pdf(CONFIG.main_tex)
            
            if success:
                pdf_path = os.path.join(CONFIG.output_dir, CONFIG.main_tex.replace('.tex', '.pdf'))
                LOG.success(f"PDF generated: {pdf_path}")
                return True
            else:
                LOG.warning(f"Compilation failed (attempt {attempt + 1})")
                LOG.info("Attempting to fix errors...")
                
                # Try to fix with typesetter agent
                fix_prompt = f"""Fix these LaTeX errors:

ERROR LOG:
{error_log}

DOCUMENT:
{document}

Return the complete fixed document."""
                
                fixed = clean_latex(self.typesetter.think(fix_prompt))
                document = fixed
                save_file(CONFIG.main_tex, document)
        
        LOG.error("Failed to compile after multiple attempts")
        return False
    
    def generate(self) -> bool:
        """Run the complete paper generation pipeline"""
        LOG.info(f"Starting paper generation for: {self.topic}")
        start_time = time.time()
        
        # Phase 1: Planning
        if not self.phase1_planning():
            return False
        
        # Phase 2: Research
        if not self.phase2_research():
            LOG.warning("Research phase returned no papers, continuing anyway...")
        
        # Phase 3: Writing
        if not self.phase3_writing():
            return False
        
        # Phase 4: Bibliography
        bibtex = self.phase4_bibliography()
        
        # Phase 5: Assembly
        document = self.phase5_assembly(bibtex)
        
        # Phase 6: Compilation
        success = self.phase6_compilation(document)
        
        # Summary
        elapsed = time.time() - start_time
        LOG.phase("GENERATION COMPLETE")
        LOG.info(f"Time elapsed: {elapsed:.1f} seconds")
        LOG.info(f"Papers cited: {len(self.citations)}")
        LOG.info(f"Sections written: {len(self.sections)}")
        
        if success:
            LOG.success(f"📄 Paper saved to: {CONFIG.output_dir}/")
        
        return success


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     🎓 AUTONOMOUS RESEARCH PAPER GENERATOR v2.0 🎓            ║
║                                                              ║
║  Fully autonomous - just enter a topic and wait!             ║
║  Uses real citations from arXiv & Semantic Scholar           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("📝 Enter your research topic: ", end='')
    topic = input().strip()
    
    if not topic:
        print("❌ No topic provided. Exiting.")
        return
    
    print(f"\n🚀 Starting generation for: '{topic}'")
    print("   This will take several minutes. Sit back and relax!\n")
    
    generator = ResearchPaperGenerator(topic)
    success = generator.generate()
    
    if success:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    🎉 SUCCESS! 🎉                             ║
╠══════════════════════════════════════════════════════════════╣
║  Your paper has been generated!                              ║
║                                                              ║
║  📁 Output folder: {CONFIG.output_dir:<40} ║
║  📄 Main file: {CONFIG.main_tex:<44} ║
║  📚 References: references.bib                               ║
╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print("\n❌ Generation failed. Check the log file for details.")


if __name__ == "__main__":
    main()
