#!/usr/bin/env python3
"""Integration check for the multimodal-SHM hardening on a live Pennyroyal engine.

Sends concurrent image chat requests to the engine while a saboteur thread
unlinks every /dev/shm/sgl_shm_mm_* segment the moment it appears, so some
requests reach the scheduler with their feature segment already gone — the
same condition as the production crashes (FileNotFoundError in __setstate__).

Expected with the fix: the engine stays up, the sabotaged requests come back
as HTTP errors (not hangs), unsabotaged requests still succeed.
Expected without the fix: the TP0 scheduler dies and the engine restarts.

Usage: shm_fault_injection.py [--url http://127.0.0.1:8001] [--n 40] [--conc 6]
"""

import argparse
import base64
import concurrent.futures as cf
import json
import os
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8001")
ap.add_argument("--model", default="pennyroyal")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--conc", type=int, default=6)
ap.add_argument(
    "--sabotage",
    type=float,
    default=1.0,
    help="fraction of time the saboteur is active",
)
ap.add_argument("--image", default="/tmp/vision_test.png")
args = ap.parse_args()

img_b64 = base64.b64encode(open(args.image, "rb").read()).decode()
stop = threading.Event()
removed = 0


def saboteur():
    global removed
    while not stop.is_set():
        try:
            for name in os.listdir("/dev/shm"):
                if name.startswith("sgl_shm_mm_"):
                    try:
                        os.unlink("/dev/shm/" + name)
                        removed += 1
                    except FileNotFoundError:
                        pass
        except OSError:
            pass
        time.sleep(0.0005)


def one(i):
    body = {
        "model": args.model,
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Request {i}: name the shapes in the image, briefly.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ],
    }
    req = urllib.request.Request(
        args.url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
            return (
                "ok",
                round(time.time() - t0, 1),
                (d["choices"][0]["message"]["content"] or "")[:40],
            )
    except urllib.error.HTTPError as e:
        return (
            "http%d" % e.code,
            round(time.time() - t0, 1),
            e.read()[:120].decode(errors="replace"),
        )
    except Exception as e:
        return ("exc", round(time.time() - t0, 1), str(e)[:120])


def health():
    try:
        with urllib.request.urlopen(args.url + "/health", timeout=5) as r:
            return r.status
    except Exception as e:
        return str(e)[:60]


print("health before:", health())
t = threading.Thread(target=saboteur, daemon=True)
if args.sabotage > 0:
    t.start()
results = []
with cf.ThreadPoolExecutor(max_workers=args.conc) as ex:
    for res in ex.map(one, range(args.n)):
        results.append(res)
        print(res, flush=True)
stop.set()
time.sleep(1)
from collections import Counter

print(
    "summary:",
    Counter(r[0] for r in results),
    "| segments removed by saboteur:",
    removed,
)
print("health after:", health())
