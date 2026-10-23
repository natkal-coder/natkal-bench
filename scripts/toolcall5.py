#!/usr/bin/env python3
"""Five tool-calling tasks through llama-server's OpenAI-compatible tools API.
Scored on whether the model emits the right function with the right arguments; the
function is never executed. Progress per task, errors first.

usage: toolcall5.py <base_url> <label> <outdir>
"""
import json, os, sys, time, urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount between currencies",
     "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "create_reminder", "description": "Create a reminder at an ISO 8601 datetime",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "when": {"type": "string"}}, "required": ["text", "when"]}}},
    {"type": "function", "function": {"name": "search_files", "description": "Search files by extension under a directory",
     "parameters": {"type": "object", "properties": {"directory": {"type": "string"}, "extension": {"type": "string"}}, "required": ["directory", "extension"]}}},
]

TASKS = [
    ("What is the weather in Lisbon right now, in celsius?", "get_weather", {"city": "lisbon", "unit": "celsius"}),
    ("Convert 250 euros to Japanese yen.", "convert_currency", {"amount": 250, "from_currency": "eur", "to_currency": "jpy"}),
    ("Remind me to call the dentist on 2026-11-03 at 09:30.", "create_reminder", {"when": "2026-11-03t09:30"}),
    ("Find all .log files under /var/log.", "search_files", {"directory": "/var/log", "extension": "log"}),
    ("Say hello. Do not call any tool.", None, {}),
]

def norm(v):
    if isinstance(v, str): return v.strip().lower().replace(" ", "")
    if isinstance(v, (int, float)): return float(v)
    return v

def score(call, want_name, want_args):
    if want_name is None: return call is None
    if call is None or call["name"] != want_name: return False
    args = call["args"]
    for k, v in want_args.items():
        got = args.get(k)
        if got is None: return False
        if k == "when": return norm(got).startswith(norm(v)[:16])
        if k == "extension": got = norm(got).lstrip(".")
        if norm(got) != norm(v): return False
    return True

def ask(base, q):
    body = json.dumps({"messages": [{"role": "user", "content": q}], "tools": TOOLS, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 2048}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t0
    m = d["choices"][0]["message"]; tc = m.get("tool_calls") or []
    call = None
    if tc:
        f = tc[0]["function"]
        try: args = json.loads(f.get("arguments") or "{}")
        except Exception: args = {"_raw": f.get("arguments")}
        call = {"name": f["name"], "args": args}
    return call, (m.get("content") or "")[:200], dt

def main():
    base, label, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    rows, errors, ok_n = [], 0, 0
    for i, (q, name, args) in enumerate(TASKS, 1):
        try:
            call, content, dt = ask(base, q); ok = score(call, name, args); ok_n += ok
            rows.append({"i": i, "want": name, "call": call, "ok": ok, "seconds": round(dt, 1), "content": content})
        except Exception as e:
            errors += 1; rows.append({"i": i, "want": name, "error": str(e)[:200]})
        print(f"[{label}] {i}/5 errors={errors} ok={ok_n} last={rows[-1]}", flush=True)
    json.dump(rows, open(os.path.join(outdir, f"{label}.tools.json"), "w"), indent=1)
    print(json.dumps({"label": label, "ok": ok_n, "n": 5, "errors": errors}), flush=True)
    if errors: sys.exit(1)

if __name__ == "__main__":
    main()
