#!/usr/bin/env python3
"""HumanEval subset, pass@1, against a llama-server /v1/chat/completions endpoint.
Each generated solution is executed with its official test in a separate `python3 -I`
subprocess inside a scratch dir with a 15 s timeout. Progress per problem, errors first.

usage: humaneval40.py <base_url> <label> <outdir> [n_problems=40] [max_tokens=2048]
"""
import gzip, json, os, re, subprocess, sys, tempfile, time, urllib.request

DATA = "/media/rickeshtn/683b3557-1768-4fda-a2f5-95248804092f/scratch/youtube_agent/human-eval/data/HumanEval.jsonl.gz"
SCRATCH = "/media/rickeshtn/683b3557-1768-4fda-a2f5-95248804092f/scratch/tmp/humaneval"

def chat(base, prompt, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content":
        "Complete the following Python function. Reply with the complete function definition only, in one ```python code block, no explanation.\n\n```python\n" + prompt + "\n```"}],
        "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=1800)); dt = time.time() - t0
    m = d["choices"][0]["message"]; content = m.get("content") or ""
    if "</think>" in content: content = content.split("</think>", 1)[-1]
    return content, d.get("usage", {}).get("completion_tokens", 0), dt

def extract(content, prompt):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", content, flags=re.S)
    code = blocks[-1] if blocks else content
    # if the model returned only a body, prepend the prompt (signature + docstring)
    if "def " not in code: code = prompt + code
    # the prompt's header (imports such as `from typing import List`) must survive a full rewrite
    header = prompt.split("def ", 1)[0]
    if header.strip() and header.strip() not in code: code = header + code
    return code

def run(code, test, entry):
    os.makedirs(SCRATCH, exist_ok=True)
    prog = code + "\n\n" + test + f"\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=SCRATCH, delete=False) as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=15, cwd=SCRATCH)
        return r.returncode == 0, (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)

def main():
    base, label, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
    os.makedirs(outdir, exist_ok=True)
    probs = [json.loads(l) for l in gzip.open(DATA, "rt")][:n]
    rows, errors, passed = [], 0, 0
    for i, p in enumerate(probs, 1):
        try:
            content, ntok, dt = chat(base, p["prompt"], max_tokens)
            code = extract(content, p["prompt"])
            ok, err = run(code, p["test"], p["entry_point"]); passed += ok
            rows.append({"task_id": p["task_id"], "passed": ok, "gen_tokens": ntok, "seconds": round(dt, 1), "err": err if not ok else "", "code": code})
        except Exception as e:
            errors += 1; rows.append({"task_id": p["task_id"], "error": str(e)[:200]})
        print(f"[{label}] {i}/{n} errors={errors} passed={passed} ({passed/i:.2f})", flush=True)
    json.dump(rows, open(os.path.join(outdir, f"{label}.humaneval.json"), "w"), indent=1)
    done = [r for r in rows if "error" not in r]
    summary = {"label": label, "n": n, "passed": passed, "pass_at_1": round(passed / n, 3), "errors": errors,
               "total_gen_tokens": sum(r["gen_tokens"] for r in done), "total_seconds": round(sum(r["seconds"] for r in done), 1)}
    json.dump(summary, open(os.path.join(outdir, f"{label}.humaneval.summary.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)
    if errors: sys.exit(1)

if __name__ == "__main__":
    main()
