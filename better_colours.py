#!/usr/bin/env python3


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

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from rich.box import ROUNDED, DOUBLE, HEAVY
from rich.style import Style
from rich.theme import Theme
from rich import print as rprint

# Custom theme for the app
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow", 
    "error": "bold red",
    "success": "bold green",
    "thinking": "italic magenta",
    "response": "white",
    "agent": "bold cyan",
    "section": "bold yellow",
    "title": "bold white on blue",
})

console = Console(theme=custom_theme, force_terminal=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Centralized configuration - modify these settings as needed"""
    # Model settings
    model: str = "qwen3-vl:latest"
    temperature: float = 0.7
    enable_thinking: bool = True  # Show model's reasoning process
    
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
    
    # Logging
    verbose_logging: bool = True  # Log all inputs/outputs
    log_thinking: bool = True  # Log model thinking process


CONFIG = Config()
os.makedirs(CONFIG.output_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    """Rich-powered logging with beautiful terminal output"""
    
    def __init__(self, output_dir: str):
        self.log_path = os.path.join(output_dir, "research.log")
        self.thinking_log_path = os.path.join(output_dir, "thinking.log")
        self.verbose_log_path = os.path.join(output_dir, "verbose.log")
        
        # Initialize log files
        for path in [self.log_path, self.thinking_log_path, self.verbose_log_path]:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*70}\n")
                f.write(f"LOG STARTED: {datetime.datetime.now()}\n")
                f.write(f"{'='*70}\n\n")
    
    def _write_file(self, path: str, msg: str):
        """Write to log file"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {msg}\n")
    
    def info(self, msg: str):
        console.print(f"  [info]ℹ️  {msg}[/info]")
        self._write_file(self.log_path, f"[INFO] {msg}")
    
    def success(self, msg: str):
        console.print(f"  [success]✅ {msg}[/success]")
        self._write_file(self.log_path, f"[SUCCESS] {msg}")
    
    def warning(self, msg: str):
        console.print(f"  [warning]⚠️  {msg}[/warning]")
        self._write_file(self.log_path, f"[WARNING] {msg}")
    
    def error(self, msg: str):
        console.print(f"  [error]❌ {msg}[/error]")
        self._write_file(self.log_path, f"[ERROR] {msg}")
    
    def phase(self, msg: str):
        """Display a major phase header"""
        console.print()
        console.rule(f"[bold blue]{msg}[/bold blue]", style="blue")
        console.print()
        self._write_file(self.log_path, f"\n{'='*50}\n{msg}\n{'='*50}")
    
    def agent(self, name: str, msg: str):
        """Log agent activity"""
        self._write_file(self.log_path, f"[AGENT:{name}] {msg}")
    
    def thinking(self, agent_name: str, thinking_content: str):
        """Log model's thinking to file"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        with open(self.thinking_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'─'*60}\n")
            f.write(f"[{timestamp}] AGENT: {agent_name}\n")
            f.write(f"{'─'*60}\n")
            f.write(f"{thinking_content}\n")
    
    def verbose_input(self, agent_name: str, prompt: str):
        """Log full input to file"""
        self._write_file(self.verbose_log_path, f"\n[INPUT:{agent_name}]\n{prompt}")
    
    def verbose_output(self, agent_name: str, response: str):
        """Log full output to file"""
        self._write_file(self.verbose_log_path, f"\n[OUTPUT:{agent_name}]\n{response}")


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
    def extract_search_terms_with_llm(topic: str) -> List[str]:
        """Use LLM to extract the best search terms from any topic"""
        
        console.print()
        console.print(Panel(
            "[bold]Extracting search keywords...[/bold]",
            title="[bold bright_green]🔍 Keyword Extractor[/bold bright_green]",
            border_style="bright_green",
            padding=(0, 2)
        ))
        
        # CONCISE prompt to prevent over-reasoning
        prompt = f"""Extract 5-7 academic search keywords from this topic.

TOPIC: {topic}

Output a JSON array: ["keyword1", "keyword2", ...]
Technical terms only. 1-3 words each. No generic words.

JSON:"""

        try:
            stream = ollama.chat(
                model=CONFIG.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3},
                think=True,
                stream=True,
            )
            
            thinking_content = ""
            content = ""
            thinking_started = False
            content_started = False
            
            for chunk in stream:
                if hasattr(chunk, 'message') and chunk.message:
                    chunk_thinking = getattr(chunk.message, 'thinking', None)
                    if chunk_thinking:
                        if not thinking_started:
                            thinking_started = True
                            console.print(f"\n  [dim]💭 Thinking:[/dim]")
                        # Full streaming of thinking
                        console.print(f"[dim italic]{chunk_thinking}[/dim italic]", end='')
                        thinking_content += chunk_thinking
                    
                    chunk_content = getattr(chunk.message, 'content', None)
                    if chunk_content:
                        if not content_started:
                            content_started = True
                            if thinking_started:
                                console.print()
                            console.print(f"\n  [bold green]📝 Keywords:[/bold green]")
                        console.print(f"[green]{chunk_content}[/green]", end='')
                        content += chunk_content
            
            console.print()
            
            # Clean up response
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = content.strip()
            
            # Try to find JSON array
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                keywords = json.loads(match.group())
                keywords = [k.strip() for k in keywords if k.strip() and len(k.strip()) > 2]
                return keywords[:7]
        except Exception as e:
            console.print(f"[warning]LLM keyword extraction failed: {e}[/warning]")
        
        # Fallback: simple extraction
        return ArxivTool.extract_search_terms_simple(topic)
    
    @staticmethod
    def extract_search_terms_simple(topic: str) -> List[str]:
        """Fallback: Extract terms without LLM"""
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', topic)
        
        # Extract capitalized terms/acronyms
        caps = re.findall(r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][a-z]+)*\b', topic)
        caps = [c for c in caps if c.lower() not in ['the', 'this', 'that', 'most', 'while', 'would', 'could', 'should']]
        
        # Extract hyphenated terms (often technical)
        hyphenated = re.findall(r'\b\w+-\w+(?:-\w+)*\b', topic)
        
        # Combine and dedupe
        keywords = list(set(quoted + caps[:5] + hyphenated[:3]))
        
        # If still nothing, use first significant words
        if len(keywords) < 3:
            words = topic.split()
            keywords.extend([w for w in words[:10] if len(w) > 4 and w.lower() not in 
                           ['would', 'could', 'should', 'their', 'there', 'these', 'those',
                            'which', 'while', 'about', 'being', 'having', 'paper', 'method',
                            'technique', 'approach', 'current', 'specific', 'without']])
        
        return list(set(keywords))[:6]
    
    @staticmethod
    @retry(max_attempts=3)
    @cache_result
    def search(query: str, max_results: int = 10, use_category_filter: bool = True) -> List[Dict]:
        """Search arXiv and return paper metadata"""
        
        # If query is too long, it's probably a full topic - extract keywords
        if len(query) > 100:
            keywords = ArxivTool.extract_search_terms_with_llm(query)
            query = ' '.join(keywords[:4])  # Use top 4 keywords
        
        # Clean up query
        query = query.strip()
        if not query:
            LOG.warning("Empty query, cannot search arXiv")
            return []
        
        LOG.info(f"arXiv search query: '{query}'")
        
        # Build search query
        if use_category_filter:
            # Try with CS categories first for better relevance
            search_query = f'all:{query} AND (cat:cs.LG OR cat:cs.AI OR cat:cs.CL OR cat:cs.CV OR cat:cs.NE)'
        else:
            search_query = f'all:{query}'
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = requests.get(ArxivTool.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        papers = ArxivTool._parse_arxiv_response(response.content)
        
        # If no results with category filter, try without
        if not papers and use_category_filter:
            LOG.warning("No results with CS filter, trying broader search...")
            return ArxivTool.search(query, max_results, use_category_filter=False)
        
        return papers
    
    @staticmethod
    def _parse_arxiv_response(content: bytes) -> List[Dict]:
        """Parse arXiv API XML response"""
        root = ET.fromstring(content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            if not authors:
                continue
                
            # Get first author's last name for citation key
            first_author_last = authors[0].split()[-1] if authors else "Unknown"
            # Clean author name for cite key
            first_author_last = re.sub(r'[^a-zA-Z]', '', first_author_last)
            
            # Extract year from published date
            published = entry.find('atom:published', ns)
            year = published.text[:4] if published is not None else "2024"
            
            title_elem = entry.find('atom:title', ns)
            summary_elem = entry.find('atom:summary', ns)
            arxiv_id = entry.find('atom:id', ns)
            
            if title_elem is None or not title_elem.text:
                continue
            
            title = title_elem.text.strip().replace('\n', ' ')
            
            # Generate citation key
            title_word = re.sub(r'[^a-zA-Z]', '', title.split()[0] if title else "paper").lower()
            cite_key = f"{first_author_last.lower()}{year}{title_word}"
            
            papers.append({
                'title': title,
                'authors': authors,
                'year': year,
                'abstract': summary_elem.text.strip().replace('\n', ' ')[:500] if summary_elem is not None and summary_elem.text else "",
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
        
        # Escape special characters in title
        title = paper['title'].replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')
        
        return f"""@article{{{paper['cite_key']},
    title={{{title}}},
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
        """Search for papers with cleaned query"""
        
        # Clean and shorten query
        if len(query) > 100:
            # Extract just the key terms
            words = query.split()
            important_words = [w for w in words if len(w) > 4 and w.lower() not in 
                             ['would', 'could', 'should', 'their', 'there', 'these', 'those', 
                              'which', 'while', 'about', 'being', 'having', 'paper', 'method']]
            query = ' '.join(important_words[:6])
        
        LOG.info(f"Semantic Scholar query: '{query}'")
        
        url = f"{SemanticScholarTool.BASE_URL}/paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,authors,year,abstract,citationCount,url,externalIds'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 429:  # Rate limited
                LOG.warning("Semantic Scholar rate limited, waiting...")
                time.sleep(5)
                response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                LOG.warning(f"Semantic Scholar returned status {response.status_code}")
                return []
            
            data = response.json()
            papers = []
            
            for paper in data.get('data', []):
                if not paper.get('title'):
                    continue
                    
                authors = [a.get('name', '') for a in paper.get('authors', [])[:5]]
                first_author_last = authors[0].split()[-1] if authors else "Unknown"
                year = str(paper.get('year', '2024') or '2024')
                
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
            
        except Exception as e:
            LOG.warning(f"Semantic Scholar search failed: {e}")
            return []


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
    _init_attempted = False
    
    @classmethod
    def get_tool(cls):
        if cls._init_attempted:
            return cls._tool
            
        cls._init_attempted = True
        try:
            import language_tool_python
            LOG.info("Initializing grammar tool (first run downloads ~1GB)...")
            cls._tool = language_tool_python.LanguageTool('en-US')
            LOG.success("Grammar tool initialized")
        except ImportError:
            LOG.warning("Grammar tool not available. Install: pip install language-tool-python")
        except Exception as e:
            LOG.warning(f"Grammar tool init failed: {e}")
        
        return cls._tool
    
    @classmethod
    def check(cls, text: str) -> Tuple[str, List[str]]:
        """Check grammar and return corrected text + issues found"""
        if not CONFIG.enable_grammar_check:
            return text, []
            
        tool = cls.get_tool()
        if tool is None:
            return text, []
        
        try:
            # Remove LaTeX commands for checking
            clean_text = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', '', text)
            clean_text = re.sub(r'[\$\{\}\[\]]', '', clean_text)
            
            matches = tool.check(clean_text)
            issues = [f"{m.ruleId}: {m.message}" for m in matches[:5]]  # Top 5 issues
            
            return text, issues
        except Exception as e:
            LOG.warning(f"Grammar check failed: {e}")
            return text, []


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
    """Enhanced agent with rich terminal output and real-time streaming"""
    
    def __init__(self, name: str, system_prompt: str, tools: List[str] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.conversation_history = []
        
        # Color mapping for different agents
        self.colors = {
            'Architect': 'blue',
            'Researcher': 'green', 
            'Scholar': 'yellow',
            'Critic': 'red',
            'Artist': 'magenta',
            'Librarian': 'cyan',
            'Typesetter': 'white',
            'Integrator': 'bright_blue',
        }
        self.color = self.colors.get(name, 'white')
    
    def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool and return results"""
        if tool_name in TOOLS:
            try:
                with console.status(f"[bold cyan]Executing {tool_name}...[/bold cyan]"):
                    result = TOOLS[tool_name](**kwargs)
                return result
            except Exception as e:
                console.print(f"  [warning]Tool {tool_name} failed: {e}[/warning]")
                return None
        return None
    
    @retry(max_attempts=CONFIG.max_retries, delay=CONFIG.retry_delay)
    def think(self, user_input: str, use_tools: bool = True) -> str:
        """Process input and generate response with full real-time streaming"""
        
        # Display agent header
        console.print()
        console.print(Panel(
            f"[bold]Processing...[/bold]",
            title=f"[bold {self.color}]🤖 {self.name}[/bold {self.color}]",
            border_style=self.color,
            padding=(0, 2)
        ))
        
        LOG.agent(self.name, f"Processing: {user_input[:80]}...")
        
        # Build context from tools if needed
        tool_context = ""
        if use_tools and self.tools:
            for tool_name in self.tools:
                if tool_name == 'arxiv_search' and 'research' in self.name.lower():
                    search_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_input)
                    query = ' '.join(search_terms[:3]) if search_terms else user_input[:50]
                    results = self._execute_tool('arxiv_search', query=query, max_results=5)
                    if results:
                        tool_context += "\n\nRELEVANT PAPERS FROM ARXIV:\n"
                        for p in results:
                            tool_context += f"- {p['title']} ({p['year']}) [cite: {p['cite_key']}]\n"
        
        # Build messages
        full_system_prompt = self.system_prompt + tool_context
        messages = [{'role': 'system', 'content': full_system_prompt}]
        messages.extend(self.conversation_history[-4:])
        messages.append({'role': 'user', 'content': user_input})
        
        LOG.verbose_input(self.name, f"SYSTEM: {full_system_prompt}\n\nUSER: {user_input}")
        
        # Stream response - FULL real-time output
        thinking_content = ""
        content = ""
        
        if CONFIG.enable_thinking:
            try:
                stream = ollama.chat(
                    model=CONFIG.model,
                    messages=messages,
                    options={'temperature': CONFIG.temperature},
                    think=True,
                    stream=True,
                )
                
                thinking_started = False
                content_started = False
                
                for chunk in stream:
                    if hasattr(chunk, 'message') and chunk.message:
                        # Stream thinking in real-time (character by character)
                        chunk_thinking = getattr(chunk.message, 'thinking', None)
                        if chunk_thinking:
                            if not thinking_started:
                                thinking_started = True
                                console.print(f"\n  [dim]💭 Thinking:[/dim]")
                            # Full streaming - print each chunk as it arrives
                            console.print(f"[dim italic]{chunk_thinking}[/dim italic]", end='')
                            thinking_content += chunk_thinking
                        
                        # Stream content in real-time
                        chunk_content = getattr(chunk.message, 'content', None)
                        if chunk_content:
                            if not content_started:
                                content_started = True
                                if thinking_started:
                                    console.print()  # New line after thinking
                                console.print(f"\n  [bold green]✍️  Response:[/bold green]")
                            console.print(chunk_content, end='')
                            content += chunk_content
                
                console.print()  # Final newline
                
                # Log thinking to file
                if thinking_content and CONFIG.log_thinking:
                    LOG.thinking(self.name, thinking_content)
                    
            except Exception as e:
                console.print(f"[warning]Streaming failed: {e}, retrying...[/warning]")
                response = ollama.chat(
                    model=CONFIG.model,
                    messages=messages,
                    options={'temperature': CONFIG.temperature},
                    stream=False,
                )
                content = response.message.content
                console.print(Panel(content[:500] + "..." if len(content) > 500 else content, 
                                   title="Response", border_style="green"))
        else:
            with console.status(f"[bold {self.color}]Generating response...[/bold {self.color}]"):
                response = ollama.chat(
                    model=CONFIG.model,
                    messages=messages,
                    options={'temperature': CONFIG.temperature},
                    think=False,
                    stream=False,
                )
                content = response.message.content
            console.print(Panel(content[:500] + "..." if len(content) > 500 else content,
                               title="Response", border_style="green"))
        
        # Log output
        LOG.verbose_output(self.name, content)
        
        # Update history
        self.conversation_history.append({'role': 'user', 'content': user_input})
        self.conversation_history.append({'role': 'assistant', 'content': content})
        if thinking_content:
            self.conversation_history[-1]['thinking'] = thinking_content
        
        return content
    
    def reset_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED AGENT PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

PROMPTS = {
    'architect': """You are a Research Architect. Create a paper structure as JSON.

OUTPUT FORMAT (strict JSON only):
{
    "title": "Formal Academic Title",
    "abstract": "150-200 word abstract...",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "sections": [
        {
            "title": "Introduction",
            "description": "What this section covers...",
            "subsections": ["Background", "Motivation", "Contributions"],
            "needs_diagram": false,
            "estimated_length": "400-500 words"
        }
    ]
}

RULES:
1. Include: Introduction, Related Work, Methodology, Results/Discussion, Conclusion
2. Mark methodology/results sections with needs_diagram: true
3. Output ONLY valid JSON, no explanations
4. Be concise in descriptions""",

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

    'critic': """You are an Academic Reviewer. Evaluate and score the section.

SCORES (0-10 each): technical_depth, clarity, citations, formatting, academic_tone

OUTPUT (JSON only):
{"status": "PASS/REVISE", "scores": {...}, "total_score": X, "major_issues": [], "feedback": "brief"}

PASS if total >= 35. Output ONLY JSON.""",

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
        """Phase 2: Gather research from real sources with LLM-guided searches"""
        LOG.phase("PHASE 2: RESEARCH & CITATION GATHERING")
        
        # Use LLM to extract optimal search keywords (streams in real-time)
        keywords = ArxivTool.extract_search_terms_with_llm(self.topic)
        
        # Display keywords
        console.print()
        console.print(Panel(
            " • ".join([f"[bold cyan]{k}[/bold cyan]" for k in keywords]),
            title="[bold green]✅ Extracted Keywords[/bold green]",
            border_style="green"
        ))
        console.print()
        
        all_arxiv_papers = []
        
        # Search with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            search_task = progress.add_task("[cyan]Searching arXiv...", total=3)
            
            # Strategy 1
            if keywords:
                query1 = ' '.join(keywords[:4])
                papers1 = ArxivTool.search(query1, max_results=10)
                all_arxiv_papers.extend(papers1)
                progress.update(search_task, advance=1, description=f"[cyan]Search 1: Found {len(papers1)} papers")
            
            # Strategy 2
            if len(keywords) > 3:
                query2 = ' '.join(keywords[2:5])
                papers2 = ArxivTool.search(query2, max_results=8)
                all_arxiv_papers.extend(papers2)
                progress.update(search_task, advance=1, description=f"[cyan]Search 2: Found {len(papers2)} papers")
            else:
                progress.update(search_task, advance=1)
            
            # Strategy 3
            if self.plan and 'title' in self.plan:
                title_keywords = ArxivTool.extract_search_terms_simple(self.plan['title'])
                if title_keywords:
                    query3 = ' '.join(title_keywords[:3])
                    papers3 = ArxivTool.search(query3, max_results=5)
                    all_arxiv_papers.extend(papers3)
                    progress.update(search_task, advance=1, description=f"[cyan]Search 3: Found {len(papers3)} papers")
            else:
                progress.update(search_task, advance=1)
        
        # Deduplicate
        seen_titles = set()
        unique_papers = []
        for p in all_arxiv_papers:
            title_lower = p['title'].lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(p)
        
        arxiv_papers = unique_papers
        
        # Search Semantic Scholar
        with console.status("[bold cyan]Searching Semantic Scholar...[/bold cyan]"):
            ss_query = ' '.join(keywords[:3]) if keywords else self.topic[:50]
            ss_papers = SemanticScholarTool.search(ss_query, limit=10)
        
        # Combine
        self.all_papers = arxiv_papers + ss_papers
        seen_titles = set()
        unique_papers = []
        for p in self.all_papers:
            title_lower = p['title'].lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(p)
        self.all_papers = unique_papers
        
        # Create citation lookup
        for p in self.all_papers:
            self.citations[p['cite_key']] = p
        
        # Display papers in a nice table
        if self.all_papers:
            papers_table = Table(
                title=f"📚 Found {len(self.all_papers)} Papers for Citation",
                box=ROUNDED,
                border_style="cyan",
                show_lines=True
            )
            papers_table.add_column("#", style="dim", width=3)
            papers_table.add_column("Citation Key", style="green", width=25)
            papers_table.add_column("Title", style="white", max_width=50)
            papers_table.add_column("Year", style="yellow", width=6)
            
            for i, p in enumerate(self.all_papers[:10], 1):
                title = p['title'][:47] + "..." if len(p['title']) > 50 else p['title']
                papers_table.add_row(str(i), p['cite_key'], title, p['year'])
            
            if len(self.all_papers) > 10:
                papers_table.add_row("...", "...", f"[dim]and {len(self.all_papers) - 10} more papers[/dim]", "")
            
            console.print()
            console.print(papers_table)
        else:
            console.print("[warning]⚠️  No papers found![/warning]")
        
        save_checkpoint({'papers': self.all_papers, 'citations': self.citations}, 'research')
        
        return len(self.all_papers) > 0
        
        return len(self.all_papers) > 0
    
    def phase3_writing(self) -> bool:
        """Phase 3: Write each section with quality control"""
        LOG.phase("PHASE 3: WRITING & PEER REVIEW")
        
        # Prepare citation context for writers
        citation_context = "AVAILABLE CITATIONS (use these exact keys):\n"
        for p in self.all_papers[:15]:
            citation_context += f"- \\cite{{{p['cite_key']}}}: {p['title'][:80]}... ({p['year']})\n"
        
        total_sections = len(self.plan['sections'])
        
        for idx, section in enumerate(self.plan['sections']):
            section_title = section['title']
            
            # Section header
            console.print()
            console.rule(f"[bold yellow]📝 Section {idx+1}/{total_sections}: {section_title}[/bold yellow]", style="yellow")
            
            passed = False
            attempts = 0
            draft = ""
            feedback = ""
            
            while not passed and attempts < CONFIG.max_rewrites:
                attempts += 1
                console.print(f"\n  [dim]Attempt {attempts}/{CONFIG.max_rewrites}[/dim]")
                
                prompt = f"""Write the '{section_title}' section.

SECTION DETAILS:
{json.dumps(section, indent=2)}

{citation_context}

{"PREVIOUS FEEDBACK TO ADDRESS: " + feedback if feedback else ""}

Write a complete, publication-ready section. Use the real citations provided above.
"""
                
                draft = clean_latex(self.scholar.think(prompt))
                
                # Check grammar
                if CONFIG.enable_grammar_check:
                    _, grammar_issues = GrammarTool.check(draft)
                    if grammar_issues:
                        console.print(f"  [warning]Grammar issues: {grammar_issues[:2]}[/warning]")
                
                # Peer review
                console.print()
                console.print("  [bold magenta]👀 Peer Review[/bold magenta]")
                
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
                        console.print(Panel(
                            f"[bold green]Score: {total_score}/50[/bold green]\nAttempt: {attempts}",
                            title="[green]✅ APPROVED[/green]",
                            border_style="green",
                            padding=(0, 2)
                        ))
                    else:
                        feedback = review.get('feedback', '')
                        major = review.get('major_issues', [])
                        console.print(Panel(
                            f"[yellow]Score: {total_score}/50[/yellow]\n{major[0][:100] if major else feedback[:100]}...",
                            title="[yellow]⚠️  REVISION NEEDED[/yellow]",
                            border_style="yellow",
                            padding=(0, 2)
                        ))
                            
                except json.JSONDecodeError:
                    console.print("  [warning]Could not parse review, passing by default[/warning]")
                    passed = True
            
            # Store section
            self.sections[section_title] = {
                'content': draft,
                'attempts': attempts
            }
            
            # Generate diagram if needed
            if section.get('needs_diagram', False):
                console.print("\n  [bold magenta]🎨 Generating Diagram[/bold magenta]")
                diagram_prompt = f"""Create a TikZ diagram for the '{section_title}' section.
                
Context: {section.get('description', '')}

Create a clear, professional diagram."""
                
                tikz_code = clean_latex(self.artist.think(diagram_prompt))
                self.sections[section_title]['diagram'] = tikz_code
        
        save_checkpoint({'sections': self.sections}, 'sections')
        
        # Summary
        console.print()
        console.print(Panel(
            f"[green]Successfully wrote {len(self.sections)} sections[/green]",
            title="[bold green]✅ Writing Complete[/bold green]",
            border_style="green"
        ))
        
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
    # Beautiful header
    console.print()
    console.print(Panel.fit(
        "[bold white]🎓 AUTONOMOUS RESEARCH PAPER GENERATOR v2.0 🎓[/bold white]\n\n"
        "[cyan]Fully autonomous - just enter a topic and wait![/cyan]\n"
        "[cyan]Uses real citations from arXiv & Semantic Scholar[/cyan]\n"
        "[cyan]LLM-powered keyword extraction for any domain[/cyan]",
        border_style="bright_blue",
        padding=(1, 4)
    ))
    console.print()
    
    # Show settings in a table
    settings_table = Table(title="⚙️  Current Settings", box=ROUNDED, border_style="dim")
    settings_table.add_column("Setting", style="cyan")
    settings_table.add_column("Value", style="green")
    
    settings_table.add_row("Model", CONFIG.model)
    settings_table.add_row("Thinking Mode", "✅ ON" if CONFIG.enable_thinking else "❌ OFF")
    settings_table.add_row("Verbose Logging", "✅ ON" if CONFIG.verbose_logging else "❌ OFF")
    settings_table.add_row("Grammar Check", "✅ ON" if CONFIG.enable_grammar_check else "❌ OFF")
    settings_table.add_row("Output Directory", f"{CONFIG.output_dir}/")
    
    console.print(settings_table)
    console.print()
    
    # Get topic input
    topic = console.input("[bold yellow]📝 Enter your research topic:[/bold yellow] ").strip()
    
    if not topic:
        console.print("[error]❌ No topic provided. Exiting.[/error]")
        return
    
    console.print()
    console.print(Panel(
        f"[bold]{topic}[/bold]",
        title="[bold green]🚀 Starting Generation[/bold green]",
        border_style="green",
        padding=(0, 2)
    ))
    console.print("[dim]This will take several minutes. Sit back and relax![/dim]\n")
    
    generator = ResearchPaperGenerator(topic)
    success = generator.generate()
    
    if success:
        # Success panel
        console.print()
        success_content = f"""[bold green]Your paper has been generated![/bold green]

[cyan]📁 Output folder:[/cyan] {CONFIG.output_dir}/
[cyan]📄 Main file:[/cyan] {CONFIG.main_tex}
[cyan]📚 References:[/cyan] references.bib

[dim]📋 Log files:[/dim]
  • research.log - Main progress log
  • thinking.log - Model reasoning/thinking  
  • verbose.log - All inputs & outputs"""
        
        console.print(Panel(
            success_content,
            title="[bold green]🎉 SUCCESS! 🎉[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
    else:
        console.print()
        console.print(Panel(
            "[bold]Generation failed. Check the log file for details.[/bold]",
            title="[bold red]❌ Error[/bold red]",
            border_style="red"
        ))


if __name__ == "__main__":
    main()
