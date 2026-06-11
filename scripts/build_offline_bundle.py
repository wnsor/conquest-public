"""
Build a single self-contained HTML file of the conquest webapp for offline viewing.

Reads webapp/index.html + app.js + styles.css, downloads Plotly once and caches
it, base64-embeds every JSON/CSV the app fetches, monkey-patches window.fetch
so the embedded code can run unchanged, and writes the result to
webapp/conquest_offline.html. The recipient double-clicks the file; everything
renders, no server, no network, no setup.

Run from workspace root:
    python scripts/build_offline_bundle.py
"""

from __future__ import annotations

import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
PARENT = WORKSPACE  # data lives in the same tree as the webapp source

WEBAPP = WORKSPACE / "webapp"
INDEX = WEBAPP / "index.html"
APP_JS = WEBAPP / "app.js"
STYLES = WEBAPP / "styles.css"

CACHE = WORKSPACE / ".cache"
PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_CACHE = CACHE / "plotly-2.35.2.min.js"

OUTPUT = WEBAPP / "conquest_offline.html"

# Manifest derived from webapp/app.js fetchJson/fetchText calls.
# Each entry: (path-as-app.js-fetches-it, parse-as) where parse is 'json' or 'text'.
# Equity curves + matching _lean_stats.json sidecars (auto-paired below).
EQUITY_CURVES = [
    "cstability",
    "cgrowth",
    "cf_25k",
    "cstag_pit",
    "cstag_voltgt_canonical",
    "rebal_60_40_50k",
    "voltgt_standalone_25k",
    "synthetic_full_strategy",
    "synthetic_voltgt_proxy",
]
DATA_MANIFEST = []
for label in EQUITY_CURVES:
    DATA_MANIFEST.append((f"storage/conquest/lean/{label}_lean_equity.json", "json"))
    DATA_MANIFEST.append((f"storage/conquest/lean/{label}_lean_stats.json",  "json"))
DATA_MANIFEST.extend([
    ("storage/conquest/lean/cstability_spy_benchmark.json", "json"),
    ("storage/conquest/lean/spy_since_inception.json",      "json"),
    ("storage/conquest/lean/qqq_buyhold_lean.json",         "json"),
    ("storage/conquest/lean/iwm_buyhold_lean.json",         "json"),
    ("storage/conquest/lean/efa_buyhold_lean.json",         "json"),
    ("storage/conquest/lean/gld_buyhold_lean.json",         "json"),
    ("storage/conquest/lean/vti_buyhold_lean.json",         "json"),
    ("storage/conquest/lean/vgt_buyhold_lean.json",         "json"),
    ("storage/conquest/lean/xlk_buyhold_lean.json",         "json"),
    ("storage/conquest/lean/vwo_buyhold_lean.json",         "json"),
    ("storage/conquest/lean/vnq_buyhold_lean.json",         "json"),
    ("storage/conquest/lean/tlt_buyhold_lean.json",         "json"),
    ("storage/conquest/regime/daily.csv",                   "text"),
    ("storage/conquest/vix/daily.csv",                      "text"),
    ("storage/conquest/macro/unrate.csv",                   "text"),
    ("storage/conquest/macro/cpi.csv",                      "text"),
    ("storage/conquest/macro/gdp.csv",                      "text"),
])


def fetch_plotly() -> str:
    if PLOTLY_CACHE.exists():
        return PLOTLY_CACHE.read_text()
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {PLOTLY_URL} (one-time)...")
    with urllib.request.urlopen(PLOTLY_URL, timeout=60) as r:
        body = r.read().decode()
    PLOTLY_CACHE.write_text(body)
    return body


def read_data_file(rel_path: str) -> bytes:
    """Resolve a data path against the parent checkout (storage/ + MODELS.json)."""
    p = PARENT / rel_path
    if not p.exists():
        raise FileNotFoundError(f"data file missing: {p}")
    return p.read_bytes()


def script_safe(code: str) -> str:
    """Escape any literal `</script>` so an inlined <script> block never closes early."""
    return code.replace("</script>", "<\\/script>")


def build() -> None:
    if not INDEX.exists() or not APP_JS.exists() or not STYLES.exists():
        sys.exit(f"ERROR: webapp source missing in {WEBAPP}")

    print(f"reading webapp sources from {WEBAPP}")
    index_html = INDEX.read_text()
    app_js     = APP_JS.read_text()
    styles_css = STYLES.read_text()

    print(f"reading data files from {PARENT}")
    data_b64: dict[str, str] = {}
    data_type: dict[str, str] = {}
    total_raw = 0
    for rel, kind in DATA_MANIFEST:
        raw = read_data_file(rel)
        total_raw += len(raw)
        data_b64[rel]  = base64.b64encode(raw).decode("ascii")
        data_type[rel] = kind
        print(f"  + {rel:60s}  {len(raw)/1024:8.1f} KB  ({kind})")
    print(f"  raw data total: {total_raw/1024/1024:.2f} MB")

    print("fetching Plotly (cached after first run)")
    plotly_js = fetch_plotly()
    print(f"  plotly: {len(plotly_js)/1024/1024:.2f} MB")

    # Build the inline-data + fetch-monkey-patch bootstrap. Runs BEFORE app.js.
    bootstrap = f"""
        // === offline-bundle bootstrap (build_offline_bundle.py) ===
        const __INLINE_DATA_B64__ = {json.dumps(data_b64)};
        const __INLINE_TYPE__     = {json.dumps(data_type)};
        function __decode_b64_utf8(b64) {{
            const bin = atob(b64);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            return new TextDecoder('utf-8').decode(bytes);
        }}
        window.__INLINE_DATA__ = {{}};
        for (const [k, b64] of Object.entries(__INLINE_DATA_B64__)) {{
            const txt = __decode_b64_utf8(b64);
            window.__INLINE_DATA__[k] = __INLINE_TYPE__[k] === 'json' ? JSON.parse(txt) : txt;
        }}
        // Monkey-patch fetch so app.js's `fetchJson`/`fetchText` resolve from
        // __INLINE_DATA__ instead of going to the network. Strips the leading
        // '../' (ROOT prefix) and any cache-busting query string.
        const __origFetch__ = window.fetch ? window.fetch.bind(window) : null;
        window.fetch = function(url, opts) {{
            const key = String(url).replace(/^\\.\\.\\//, '').replace(/\\?.*$/, '');
            if (Object.prototype.hasOwnProperty.call(window.__INLINE_DATA__, key)) {{
                const v = window.__INLINE_DATA__[key];
                const isObj = typeof v === 'object' && v !== null;
                return Promise.resolve({{
                    ok: true, status: 200, statusText: 'OK',
                    json: () => Promise.resolve(isObj ? v : JSON.parse(v)),
                    text: () => Promise.resolve(isObj ? JSON.stringify(v) : v),
                }});
            }}
            if (__origFetch__) return __origFetch__(url, opts);
            return Promise.reject(new Error('offline bundle: no fetch fallback for ' + url));
        }};
        // === end bootstrap ===
    """

    # Use lambda replacements so re.sub doesn't try to interpret \1/\g<>/etc.
    # in the replacement strings (Plotly's minified source contains literal
    # backslash sequences that would otherwise raise `bad escape`).
    style_block = f"<style>\n{styles_css}\n</style>"
    new_html = re.sub(
        r'<link[^>]*rel=["\']stylesheet["\'][^>]*href=["\']styles\.css[^"\']*["\'][^>]*>',
        lambda m: style_block,
        index_html,
        count=1,
    )

    plotly_block = f"<script>{script_safe(plotly_js)}</script>\n<script>{script_safe(bootstrap)}</script>"
    new_html = re.sub(
        r'<script[^>]*src=["\']https://cdn\.plot\.ly/[^"\']+["\'][^>]*></script>',
        lambda m: plotly_block,
        new_html,
        count=1,
    )

    appjs_block = f"<script>{script_safe(app_js)}</script>"
    new_html = re.sub(
        r'<script[^>]*src=["\']app\.js[^"\']*["\'][^>]*></script>',
        lambda m: appjs_block,
        new_html,
        count=1,
    )

    OUTPUT.write_text(new_html)
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print()
    print(f"wrote {OUTPUT}  ({size_mb:.2f} MB)")
    print(f"open with:   open {OUTPUT.relative_to(WORKSPACE)}")


if __name__ == "__main__":
    build()
