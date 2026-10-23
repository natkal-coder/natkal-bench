#!/usr/bin/env python3
"""Fixed 20-problem task with checkable numeric answers, run against a llama-server
(OpenAI-compatible) endpoint. Records per problem: correct, generation tokens, wall time,
thinking length, and the "wait benchmark": count of "wait" (case-insensitive, as a word)
in the thinking trace. Stdlib only. Prints progress per problem with errors first.

usage: task20.py <base_url> <label> <outdir> [max_tokens]
"""
import json, os, re, sys, time, urllib.request

PROBLEMS = [
    ("A shop sells pens at 3 for $4. How many dollars do 27 pens cost?", 36),
    ("A train leaves at 9:40 and arrives at 13:05 the same day. How many minutes is the trip?", 205),
    ("If 5 workers build a wall in 12 days, how many days do 4 workers need at the same rate?", 15),
    ("What is the sum of all integers from 1 to 60?", 1830),
    ("A rectangle has perimeter 50 and one side 9. What is its area?", 144),
    ("A tank holds 240 litres and leaks 15 litres per hour while a tap adds 9 per hour. After how many hours is it empty, starting full?", 40),
    ("What is 17 squared minus 13 squared?", 120),
    ("A price rises 20% then falls 25%. The final price is what percent of the original? Answer as a number.", 90),
    ("How many three-digit numbers are divisible by 7?", 128),
    ("A car averages 48 km/h for 2.5 hours then 72 km/h for 1.5 hours. Total distance in km?", 228),
    ("The mean of five numbers is 14. Four of them are 9, 12, 16 and 20. What is the fifth?", 13),
    ("What is the remainder when 2 to the power 20 is divided by 7?", 4),
    ("A box has 6 red and 4 blue balls. Two are drawn without replacement. The probability both are red, as a fraction a/b in lowest terms, has what value of a+b?", 4),
    ("How many diagonals does a convex polygon with 12 sides have?", 54),
    ("The product of two consecutive even integers is 528. What is the smaller one?", 22),
    ("A recipe for 8 people needs 600 g of flour. How many grams for 14 people?", 1050),
    ("What is the least common multiple of 12, 18 and 20?", 180),
    ("A clock shows 3:00. How many degrees is the smaller angle between the hands?", 90),
    ("Solve for x: 3(x - 4) + 2x = 23. What is x?", 7),
    ("A water bill is $18 fixed plus $2.40 per cubic metre. The bill is $78. How many cubic metres?", 25),
]

def ask(base, q, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": q + " Give the final answer as a single integer on the last line in the form ANSWER: <number>."}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=1800)); dt = time.time() - t0
    m = d["choices"][0]["message"]
    think = m.get("reasoning_content") or m.get("reasoning") or ""
    content = m.get("content") or ""
    if not think and "<think>" in content:
        think = content.split("<think>", 1)[1].split("</think>", 1)[0]; content = content.split("</think>", 1)[-1]
    usage = d.get("usage", {})
    return think, content, usage.get("completion_tokens", 0), dt

def main():
    base, label, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    os.makedirs(outdir, exist_ok=True)
    rows, errors, correct = [], 0, 0
    for i, (q, gold) in enumerate(PROBLEMS, 1):
        try:
            think, content, ntok, dt = ask(base, q, max_tokens)
            m = re.findall(r"ANSWER:\s*(-?\d+)", content) or re.findall(r"(-?\d+)\s*$", content.strip())
            got = int(m[-1]) if m else None
            ok = got == gold; correct += ok
            waits = len(re.findall(r"\bwait\b", think, flags=re.I))
            rows.append({"i": i, "gold": gold, "got": got, "correct": ok, "gen_tokens": ntok, "seconds": round(dt, 1),
                         "think_chars": len(think), "waits": waits, "tok_per_s": round(ntok / dt, 2) if dt else None})
        except Exception as e:
            errors += 1; rows.append({"i": i, "gold": gold, "error": str(e)[:200]})
        print(f"[{label}] {i}/20 errors={errors} correct={correct} last={rows[-1]}", flush=True)
    json.dump(rows, open(os.path.join(outdir, f"{label}.json"), "w"), indent=1)
    done = [r for r in rows if "error" not in r]
    tot_tok = sum(r["gen_tokens"] for r in done); tot_s = sum(r["seconds"] for r in done)
    summary = {"label": label, "correct": correct, "errors": errors, "n": len(PROBLEMS), "total_gen_tokens": tot_tok,
               "total_seconds": round(tot_s, 1), "avg_tok_per_s": round(tot_tok / tot_s, 2) if tot_s else None,
               "total_waits": sum(r["waits"] for r in done), "waits_per_problem": round(sum(r["waits"] for r in done) / max(1, len(done)), 2)}
    json.dump(summary, open(os.path.join(outdir, f"{label}.summary.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)
    if errors: sys.exit(1)

if __name__ == "__main__":
    main()
