import asyncio
import json
import os
import sys
import shutil
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def load_local_env_file():
    """Parses the literal '.env' configuration properties cleanly into context space."""
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    val = val.strip("'\"")
                    os.environ[key.strip()] = val

load_local_env_file()

class GitHubMCPIssueMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        self.last_error = None
        self.last_source = "None"

    def _generate_dynamic_fallback_issues(self, domain: str, complexity: str) -> list:
        label_map = {
            "Good First Issue (Junior)": "good first issue, documentation",
            "Help Wanted (Intermediate)": "help wanted, enhancement",
            "Core Refactor (Senior)": "core, architectural-redesign, high-priority"
        }
        labels = label_map.get(complexity, "help wanted")
        
        if domain == "Backend Engineering":
            return [
                {"id": 101, "repo": "pallets/flask", "title": "Fix memory leak in asynchronous context local tearing lifecycle hooks", "labels": labels, "difficulty": complexity, "url": "https://github.com/pallets/flask/issues/101"},
                {"id": 102, "repo": "tiangolo/fastapi", "title": "Implement automated validation type guards on heavily nested Pydantic payloads", "labels": labels, "difficulty": complexity, "url": "https://github.com/tiangolo/fastapi/issues/102"},
                {"id": 103, "repo": "django/django", "title": "Refactor multi-statement connection pooling thread locks for high throughput", "labels": labels, "difficulty": complexity, "url": "https://github.com/django/django/issues/103"}
            ]
        elif domain == "Machine Learning Ops":
            return [
                {"id": 201, "repo": "huggingface/transformers", "title": "Optimize Tensor loading precision mismatch anomalies during local pipeline execution", "labels": labels, "difficulty": complexity, "url": "https://github.com/huggingface/transformers/issues/201"},
                {"id": 202, "repo": "pytorch/pytorch", "title": "Implement accelerated similarity matrix vector dot-product kernels on Windows architectures", "labels": labels, "difficulty": complexity, "url": "https://github.com/pytorch/pytorch/issues/202"},
                {"id": 203, "repo": "meta-llama/llama", "title": "Fix token serialization tracking bugs within localized embeddings generation maps", "labels": labels, "difficulty": complexity, "url": "https://github.com/meta-llama/llama/issues/203"}
            ]
        elif domain == "Frontend Architecture":
            return [
                {"id": 301, "repo": "facebook/react", "title": "Resolve layout reflow stutter delays inside dynamic server-rendered UI card metrics", "labels": labels, "difficulty": complexity, "url": "https://github.com/facebook/react/issues/301"},
                {"id": 302, "repo": "vercel/next.js", "title": "Fix interactive data dashboard route parameter parsing edge cases", "labels": labels, "difficulty": complexity, "url": "https://github.com/vercel/next.js/issues/302"},
                {"id": 303, "repo": "streamlit/streamlit", "title": "Add customizable global component theme wrappers for enterprise analytical displays", "labels": labels, "difficulty": complexity, "url": "https://github.com/streamlit/streamlit/issues/303"}
            ]
        else:
            return [
                {"id": 401, "repo": "kubernetes/kubernetes", "title": "Fix localized container runtime secret decryption pipeline bottlenecks", "labels": labels, "difficulty": complexity, "url": "https://github.com/kubernetes/kubernetes/issues/401"},
                {"id": 402, "repo": "hashicorp/terraform", "title": "Refactor automated deployment state locks validation engine wrapper", "labels": labels, "difficulty": complexity, "url": "https://github.com/hashicorp/terraform/issues/402"},
                {"id": 403, "repo": "docker/cli", "title": "Implement defensive type guards guarding system context environment loads", "labels": labels, "difficulty": complexity, "url": "https://github.com/docker/cli/issues/403"}
            ]

    async def fetch_live_issues(self, owner: str, repo: str, max_issues: int = 20) -> list:
        self.token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        if not self.token:
            self.last_error = "Token missing."
            return []
            
        npx_path = shutil.which("npx") or shutil.which("npx.cmd") or "npx.cmd"
        server_cmd = [npx_path, "-y", "@modelcontextprotocol/server-github"]
        env = os.environ.copy()
        env["GITHUB_PERSONAL_ACCESS_TOKEN"] = self.token
        
        try:
            process = await asyncio.create_subprocess_exec(
                *server_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                shell=(sys.platform == "win32"),
                limit=1024 * 1024 * 10  
            )
            
            init_req = {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "OS-Matcher-Client", "version": "1.0.0"}
                }
            }
            process.stdin.write(json.dumps(init_req).encode('utf-8') + b'\n')
            await process.stdin.drain()
            await asyncio.sleep(1.0)
            
            tool_call = {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "list_issues",
                    "arguments": {"owner": owner, "repo": repo}
                }
            }
            process.stdin.write(json.dumps(tool_call).encode('utf-8') + b'\n')
            await process.stdin.drain()
            
            try:
                resp_bytes = await asyncio.wait_for(process.stdout.readline(), timeout=3.0)
                response_data = json.loads(resp_bytes.decode('utf-8', errors='ignore'))
                self.last_source = "GitHub MCP Server"
            except Exception:
                resp_bytes = await process.stdout.read(1024 * 1024 * 5)
                response_data = json.loads(resp_bytes.decode('utf-8', errors='ignore'))
                self.last_source = "GitHub MCP Server"
                
            process.terminate()
            await process.wait()
            
            result_block = response_data.get("result", {})
            content_list = result_block.get("content", [{}])
            raw_text = content_list[0].get("text", "[]") if content_list else "[]"
            
            raw_issues = json.loads(raw_text)[:max_issues]
            
            formatted_issues = []
            for issue in raw_issues:
                labels = ", ".join([l.get("name", "") for l in issue.get("labels", []) if isinstance(l, dict)])
                formatted_issues.append({
                    "id": issue.get("number"),
                    "repo": f"{owner}/{repo}",
                    "title": issue.get("title", ""),
                    "labels": labels if labels else "None",
                    "difficulty": "Assigned via Live Stream Labels",
                    "url": issue.get("html_url", "")
                })
            return formatted_issues
            
        except Exception:
            # ── HARDENED UNIVERSAL REST FALLBACK ROUTE FOR ALL PUBLIC REPOS ──
            import urllib.request
            try:
                url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=open&per_page={max_issues}"
                req = urllib.request.Request(url)
                req.add_header("Authorization", f"token {self.token}")
                req.add_header("User-Agent", "Python-URLLIB")
                
                with urllib.request.urlopen(req) as response:
                    res = json.loads(response.read().decode('utf-8'))
                    formatted_issues = []
                    for issue in res:
                        if "pull_request" in issue:
                            continue
                        labels = ", ".join([l.get("name", "") for l in issue.get("labels", [])])
                        formatted_issues.append({
                            "id": issue.get("number"),
                            "repo": f"{owner}/{repo}",
                            "title": issue.get("title", ""),
                            "labels": labels if labels else "None",
                            "difficulty": "Assigned via REST API Labels",
                            "url": issue.get("html_url", "")
                        })
                    self.last_source = "GitHub REST fallback"
                    return formatted_issues
            except Exception as rest_err:
                self.last_error = f"GitHub could not search {owner}/{repo}. Check that the repository exists and your token can view it."
                return []

    def compute_semantic_matches(self, developer_skills: str, issues: list, complexity: str) -> pd.DataFrame:
        if not issues:
            return pd.DataFrame()
            
        df = pd.DataFrame(issues)
        skill_embedding = self.model.encode([developer_skills])[0]
        issue_texts = (df["title"] + " " + df["labels"]).tolist()
        issue_embeddings = self.model.encode(issue_texts)
        
        similarities = []
        for emb in issue_embeddings:
            sim = np.dot(skill_embedding, emb) / (np.linalg.norm(skill_embedding) * np.linalg.norm(emb))
            similarities.append(float(sim))
            
        df["semantic_score"] = similarities
        df["difficulty_score"] = 0.5  
        df["match_score"] = df["semantic_score"]
        
        if "difficulty" not in df.columns:
            df["difficulty"] = complexity
            
        return df.sort_values(by="match_score", ascending=False).reset_index(drop=True)