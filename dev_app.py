#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DM Badcases 可视化编辑器（精简版：只保留 ORI + Binary + Sync）

新增功能：
- ✅ 二值图颜色反转（invert）：前端勾选后，后端返回 binary=255-binary，并在保存时也以反转后的 binary 生成 binary_raw/binary_rect
"""

import os
import json
import base64
import shutil
import cv2
import numpy as np
from flask import Flask, jsonify, request

# ================== 配置区 ==================
BADCASES_DIR = "/home/cat/workspace/DMCODE/SNcode/badcases"
BASE_DIR = os.path.dirname(BADCASES_DIR)
CHECKED_DIR = os.path.join(BASE_DIR, "checkedcases")
os.makedirs(CHECKED_DIR, exist_ok=True)

CELL_SIZE_PX = 16
SYNC_CELL_WIDTH = 5
BORDER_WIDTH = 10

# OTSU + 形态学默认参数（给“只有 ori”用）
MORPH_KERNEL = (3, 3)
MORPH_ITERS = 1
MORPH_OP = cv2.MORPH_CLOSE
# ==========================================

app = Flask(__name__)


# -------------------- 工具函数 --------------------
def _infer_mime(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext in [".png"]:
        return "image/png"
    return "application/octet-stream"


def file_to_data_url(path: str):
    if not path or not os.path.exists(path):
        return None
    mime = _infer_mime(path)
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


def gray_img_to_data_url(gray: np.ndarray, fmt: str = ".png"):
    if gray is None:
        return None
    if fmt.lower() not in [".png", ".jpg", ".jpeg"]:
        fmt = ".png"

    ok, buf = cv2.imencode(fmt, gray)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/png" if fmt.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def ensure_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def otsu_and_morph(gray: np.ndarray) -> np.ndarray:
    gray = ensure_gray(gray)
    if gray is None:
        return None
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones(MORPH_KERNEL, dtype=np.uint8)
    binary = cv2.morphologyEx(binary, MORPH_OP, kernel, iterations=MORPH_ITERS)
    return binary


def invert_binary(binary: np.ndarray, invert: bool) -> np.ndarray:
    """✅ NEW: 二值图反转（0<->255）"""
    if binary is None:
        return None
    if not invert:
        return binary
    return 255 - binary


def dm_array_to_image(dm_array, cell_w=SYNC_CELL_WIDTH, border=BORDER_WIDTH):
    rows, cols = dm_array.shape
    h = rows * cell_w + 2 * border
    w = cols * cell_w + 2 * border
    img = np.ones((h, w), dtype=np.uint8) * 255
    for i in range(rows):
        for j in range(cols):
            if dm_array[i, j] == 0:
                y1 = border + i * cell_w
                y2 = border + (i + 1) * cell_w
                x1 = border + j * cell_w
                x2 = border + (j + 1) * cell_w
                img[y1:y2, x1:x2] = 0
    return img


def draw_grid_overlay_on_binary(binary_gray: np.ndarray, rows: int, cols: int,
                                color=(0, 255, 0), thickness=1) -> np.ndarray:
    if binary_gray is None:
        return None
    h, w = binary_gray.shape[:2]
    bgr = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2BGR)

    for i in range(rows + 1):
        y = int(round(i * h / rows))
        y = min(max(y, 0), h - 1)
        cv2.line(bgr, (0, y), (w - 1, y), color, thickness)

    for j in range(cols + 1):
        x = int(round(j * w / cols))
        x = min(max(x, 0), w - 1)
        cv2.line(bgr, (x, 0), (x, h - 1), color, thickness)

    return bgr


def parse_bool_param(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


# -------------------- Case 扫描逻辑 --------------------
def list_cases():
    files = sorted(os.listdir(BADCASES_DIR))
    info = {}

    def ensure(prefix: str):
        if prefix not in info:
            info[prefix] = {
                "prefix": prefix,
                "has_sync": False,
                "sync_path": None,
                "ori_path": None,
                "ori_pad10_path": None,
                "dm_image_path": None,
                "binary_raw_path": None,
                "binary_rect_path": None,
            }

    for f in files:
        if f.endswith("_sync_dm_array.npy"):
            prefix = f[:-len("_sync_dm_array.npy")]
            ensure(prefix)
            info[prefix]["has_sync"] = True
            info[prefix]["sync_path"] = os.path.join(BADCASES_DIR, f)
        elif f.endswith("_ori.png"):
            prefix = f[:-len("_ori.png")]
            ensure(prefix)
            info[prefix]["ori_path"] = os.path.join(BADCASES_DIR, f)
        elif f.endswith("_ori_pad10.png"):
            prefix = f[:-len("_ori_pad10.png")]
            ensure(prefix)
            info[prefix]["ori_pad10_path"] = os.path.join(BADCASES_DIR, f)

    for f in files:
        for prefix in list(info.keys()):
            if not f.startswith(prefix + "_"):
                continue
            p = os.path.join(BADCASES_DIR, f)
            if "_dm_image_" in f:
                info[prefix]["dm_image_path"] = p
            elif f.startswith(prefix + "_binary_dm_image_rect_"):
                info[prefix]["binary_rect_path"] = p
            elif f.startswith(prefix + "_binary_dm_image_") and "_binary_dm_image_rect_" not in f:
                info[prefix]["binary_raw_path"] = p

    cases = []
    for prefix, c in info.items():
        show_ori = c["ori_path"] or c["ori_pad10_path"] or c["dm_image_path"]
        c["show_ori_path"] = show_ori
        cases.append(c)

    def key_fn(c):
        try:
            return int(c["prefix"].split("_", 1)[0])
        except Exception:
            return 10**18

    return sorted(cases, key=key_fn)


def parse_case_id(prefix: str):
    try:
        return int(prefix.split("_", 1)[0])
    except Exception:
        return None


# -------------------- Flask 路由 --------------------
@app.route("/")
def index():
    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>DM Badcases Editor (ORI + Binary + Sync)</title>
  <style>
    :root {{
      --bg: #050916;
      --panel: #0b1020;
      --text: #e8eefc;
      --sub: #9aa4bf;
      --accent: #4fd1c5;
      --warn: #ffd93b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 22px 26px 30px;
      background: radial-gradient(circle at top, #101a33 0, #050916 55%, #02040a 100%);
      color: var(--text);
      font-family: -apple-system,BlinkMacSystemFont,system-ui,sans-serif;
    }}
    #app {{ max-width: 1280px; margin: 0 auto; }}
    #controls {{
      display:flex; flex-wrap:wrap; align-items:center; gap:10px;
      margin-bottom: 10px;
    }}
    .btn {{
      padding: 7px 14px; border-radius:999px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(11,16,32,0.92);
      color: var(--text);
      cursor: pointer;
      font-size: 13px;
    }}
    .btn.primary {{
      background: radial-gradient(circle at top left, rgba(79,209,197,0.22), rgba(11,16,32,0.95));
      border-color: rgba(79,209,197,0.35);
    }}
    .btn:hover {{ border-color: var(--accent); }}
    .meta {{ color: var(--sub); font-size: 12px; }}
    .hint {{ color: var(--warn); font-size: 12px; }}
    .input {{
      width: 88px;
      padding: 6px 10px;
      border-radius:999px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(6,10,22,0.6);
      color: var(--text);
      outline:none;
    }}
    .toggle {{
      display:inline-flex; align-items:center; gap:6px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(6,10,22,0.55);
      color: var(--sub);
      font-size: 12px;
    }}
    .toggle input {{ accent-color: var(--accent); }}

    .grid {{
      display:grid;
      grid-template-columns: 360px 1fr 420px;
      gap: 16px;
      margin-top: 12px;
      align-items:start;
    }}
    .panel {{
      background: radial-gradient(circle at top left, rgba(76,141,245,0.08), var(--panel));
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.05);
      box-shadow: 0 18px 40px rgba(0,0,0,0.60);
      padding: 12px 12px 14px;
    }}
    .title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.22em;
      color: var(--sub);
      margin: 2px 6px 10px;
    }}
    img {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.06);
      background: #050814;
    }}
    canvas {{
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.06);
      background: #050814;
      display:block;
      margin: 0 auto;
    }}
    #info {{
      margin-top: 6px;
      min-height: 18px;
      color: var(--accent);
      font-size: 13px;
    }}
  </style>
</head>
<body>
<div id="app">
  <div id="controls">
    <button id="prevBtn" class="btn">Prev</button>
    <button id="nextBtn" class="btn">Next</button>

    <span id="indexInfo" class="meta"></span>

    <span class="meta">rows:</span>
    <input id="rowsInput" type="number" class="input" placeholder="例如 22" />
    <span class="meta">cols:</span>
    <input id="colsInput" type="number" class="input" placeholder="例如 22" />

    <button id="initBtn" class="btn">Init Blank Sync</button>
    <button id="saveBtn" class="btn primary">Save to checkedcases</button>

    <!-- ✅ NEW: invert toggle -->
    <label class="toggle">
      <input type="checkbox" id="invertToggle" />
      <span>反转二值图</span>
    </label>

    <span id="modeHint" class="hint"></span>
  </div>

  <div id="info"></div>

  <div class="grid">
    <div class="panel">
      <div class="title">ORI</div>
      <img id="oriImg" src="" alt="ori"/>
      <div class="meta" id="oriMeta" style="margin-top:8px;"></div>
    </div>

    <div class="panel">
      <div class="title">BINARY (OTSU + MORPH) + GRID</div>
      <canvas id="binaryCanvas"></canvas>
      <div class="meta" id="binaryMeta" style="margin-top:8px;"></div>
    </div>

    <div class="panel">
      <div class="title">SYNC (CLICK TO DRAW) · 1=WHITE 0=BLACK</div>
      <canvas id="syncCanvas"></canvas>
      <div class="meta" id="syncMeta" style="margin-top:8px;"></div>
    </div>
  </div>
</div>

<script>
  const CELL_SIZE = {CELL_SIZE_PX};
  const HIGHLIGHT = "#ffd93b";

  let allCases = [];
  let currentIndex = 0;

  let prefix = "";
  let hasSync = false;

  let rows = 0;
  let cols = 0;
  let syncArray = [];

  let oriDataUrl = null;
  let binaryDataUrl = null;
  let oriW = null, oriH = null;
  let binW = null, binH = null;

  let hiR = null, hiC = null;

  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const saveBtn = document.getElementById("saveBtn");
  const initBtn = document.getElementById("initBtn");

  const rowsInput = document.getElementById("rowsInput");
  const colsInput = document.getElementById("colsInput");

  const indexInfo = document.getElementById("indexInfo");
  const infoDiv = document.getElementById("info");
  const modeHint = document.getElementById("modeHint");

  const oriImg = document.getElementById("oriImg");
  const oriMeta = document.getElementById("oriMeta");

  const binaryCanvas = document.getElementById("binaryCanvas");
  const binaryCtx = binaryCanvas.getContext("2d");
  const binaryMeta = document.getElementById("binaryMeta");

  const syncCanvas = document.getElementById("syncCanvas");
  const syncCtx = syncCanvas.getContext("2d");
  const syncMeta = document.getElementById("syncMeta");

  // ✅ NEW
  const invertToggle = document.getElementById("invertToggle");

  function fetchCases() {{
    fetch("/api/cases")
      .then(r => r.json())
      .then(data => {{
        allCases = data.cases || [];
        if (!allCases.length) {{
          infoDiv.textContent = "badcases 目录下未找到可用样本（至少需要 *_ori.png 或 *_sync_dm_array.npy）";
          return;
        }}
        loadCase(0);
      }})
      .catch(err => {{
        console.error(err);
        infoDiv.textContent = "加载 cases 出错";
      }});
  }}

  function loadCase(idx) {{
    if (!allCases.length) return;
    if (idx < 0 || idx >= allCases.length) return;
    currentIndex = idx;

    const c = allCases[currentIndex];
    const serverIndex = c.index;

    const inv = invertToggle.checked ? 1 : 0; // ✅ NEW
    fetch(`/api/case/${{serverIndex}}?invert=${{inv}}`)
      .then(r => r.json())
      .then(data => {{
        prefix = data.prefix;
        hasSync = data.has_sync;

        oriDataUrl = data.ori_data_url;
        binaryDataUrl = data.binary_data_url;

        oriW = data.ori_w; oriH = data.ori_h;
        binW = data.binary_w; binH = data.binary_h;

        rows = data.rows || 0;
        cols = data.cols || 0;
        syncArray = data.sync_array || [];

        hiR = null; hiC = null;

        const idText = (c.case_id != null) ? ` · ID: ${{c.case_id}}` : "";
        indexInfo.textContent = `当前第 ${{currentIndex+1}} / ${{allCases.length}} 个 · prefix: ${{prefix}}${{idText}}`;

        oriImg.src = oriDataUrl || "";
        oriMeta.textContent = oriW && oriH ? `ori: ${{oriW}}×${{oriH}}` : "";

        rowsInput.value = rows ? rows : "";
        colsInput.value = cols ? cols : "";

        if (!hasSync) {{
          modeHint.textContent = "（该样本无 sync_npy：请先输入 rows/cols，然后点击 Init Blank Sync）";
        }} else {{
          modeHint.textContent = "";
        }}

        drawAll();
        infoDiv.textContent = "";
      }})
      .catch(err => {{
        console.error(err);
        infoDiv.textContent = "加载 case 失败";
      }});
  }}

  function drawGrid(ctx, r, c, cellSize, color="rgba(0,255,128,0.70)", alpha=0.7) {{
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 0.6;

    for (let i=0; i<=r; i++) {{
      const y = i * cellSize + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(c * cellSize, y);
      ctx.stroke();
    }}
    for (let j=0; j<=c; j++) {{
      const x = j * cellSize + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, r * cellSize);
      ctx.stroke();
    }}
    ctx.restore();
  }}

  function drawSync() {{
    if (!rows || !cols || !syncArray.length) {{
      syncCanvas.width = 2;
      syncCanvas.height = 2;
      syncCtx.clearRect(0,0,2,2);
      syncMeta.textContent = "sync: (未初始化)";
      return;
    }}
    syncCanvas.width = cols * CELL_SIZE;
    syncCanvas.height = rows * CELL_SIZE;

    syncCtx.clearRect(0,0,syncCanvas.width,syncCanvas.height);
    for (let i=0; i<rows; i++) {{
      for (let j=0; j<cols; j++) {{
        const v = syncArray[i][j];
        syncCtx.fillStyle = v ? "#ffffff" : "#050816";
        syncCtx.fillRect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE);
      }}
    }}
    drawGrid(syncCtx, rows, cols, CELL_SIZE, "rgba(255,255,255,0.35)", 0.45);

    if (hiR !== null && hiC !== null) {{
      syncCtx.save();
      syncCtx.strokeStyle = HIGHLIGHT;
      syncCtx.lineWidth = 2;
      syncCtx.shadowColor = HIGHLIGHT;
      syncCtx.shadowBlur = 12;
      syncCtx.strokeRect(hiC*CELL_SIZE+1, hiR*CELL_SIZE+1, CELL_SIZE-2, CELL_SIZE-2);
      syncCtx.restore();
    }}

    syncMeta.textContent = `sync: ${{rows}}×${{cols}}`;
  }}

  function drawBinary() {{
    if (!rows || !cols) {{
      binaryCanvas.width = 2;
      binaryCanvas.height = 2;
      binaryCtx.clearRect(0,0,2,2);
      binaryMeta.textContent = binW && binH ? `binary: ${{binW}}×${{binH}}（请先输入 rows/cols）` : "binary: -";
      return;
    }}

    const w = cols * CELL_SIZE;
    const h = rows * CELL_SIZE;
    binaryCanvas.width = w;
    binaryCanvas.height = h;

    binaryCtx.clearRect(0,0,w,h);
    binaryCtx.fillStyle = "rgba(0,0,0,0.85)";
    binaryCtx.fillRect(0,0,w,h);

    if (binaryDataUrl) {{
      const img = new Image();
      img.onload = () => {{
        binaryCtx.save();
        binaryCtx.globalAlpha = 0.92;
        binaryCtx.drawImage(img, 0, 0, w, h);
        binaryCtx.restore();

        drawGrid(binaryCtx, rows, cols, CELL_SIZE, "rgba(0,255,128,0.85)", 0.8);

        if (hiR !== null && hiC !== null) {{
          binaryCtx.save();
          binaryCtx.strokeStyle = HIGHLIGHT;
          binaryCtx.lineWidth = 2;
          binaryCtx.shadowColor = HIGHLIGHT;
          binaryCtx.shadowBlur = 12;
          binaryCtx.strokeRect(hiC*CELL_SIZE+1, hiR*CELL_SIZE+1, CELL_SIZE-2, CELL_SIZE-2);
          binaryCtx.restore();
        }}
      }};
      img.src = binaryDataUrl;
    }} else {{
      drawGrid(binaryCtx, rows, cols, CELL_SIZE, "rgba(0,255,128,0.65)", 0.7);
    }}

    binaryMeta.textContent = binW && binH ? `binary: ${{binW}}×${{binH}} · grid: ${{rows}}×${{cols}}` : `grid: ${{rows}}×${{cols}}`;
  }}

  function drawAll() {{
    drawSync();
    drawBinary();
  }}

  function toggleCell(arr, x, y) {{
    const j = Math.floor(x / CELL_SIZE);
    const i = Math.floor(y / CELL_SIZE);
    if (i<0 || i>=rows || j<0 || j>=cols) return null;
    arr[i][j] = arr[i][j] ? 0 : 1;
    return {{i, j}};
  }}

  syncCanvas.addEventListener("click", (e) => {{
    if (!rows || !cols || !syncArray.length) return;
    const rect = syncCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const idx = toggleCell(syncArray, x, y);
    if (idx) {{
      hiR = idx.i; hiC = idx.j;
      drawAll();
    }}
  }});

  prevBtn.addEventListener("click", () => {{
    if (!allCases.length) return;
    const idx = (currentIndex - 1 + allCases.length) % allCases.length;
    loadCase(idx);
  }});

  nextBtn.addEventListener("click", () => {{
    if (!allCases.length) return;
    const idx = (currentIndex + 1) % allCases.length;
    loadCase(idx);
  }});

  initBtn.addEventListener("click", () => {{
    const r = parseInt(rowsInput.value);
    const c = parseInt(colsInput.value);
    if (!Number.isFinite(r) || !Number.isFinite(c) || r<=0 || c<=0) {{
      infoDiv.textContent = "rows/cols 输入不合法";
      return;
    }}

    fetch("/api/init_sync", {{
      method: "POST",
      headers: {{ "Content-Type":"application/json" }},
      body: JSON.stringify({{
        prefix: prefix,
        rows: r,
        cols: c
      }})
    }})
    .then(rsp => rsp.json())
    .then(data => {{
      if (data.status !== "ok") {{
        infoDiv.textContent = "Init 失败: " + (data.msg || "");
        return;
      }}
      rows = data.rows;
      cols = data.cols;
      syncArray = data.sync_array;
      hasSync = true;
      hiR = null; hiC = null;
      infoDiv.textContent = "已初始化空白 sync，可开始画图";
      drawAll();
    }})
    .catch(err => {{
      console.error(err);
      infoDiv.textContent = "Init 出错";
    }});
  }});

  saveBtn.addEventListener("click", () => {{
    if (!prefix) return;
    if (!rows || !cols || !syncArray.length) {{
      infoDiv.textContent = "请先 Init（或加载已有 sync）";
      return;
    }}

    fetch("/api/save", {{
      method: "POST",
      headers: {{ "Content-Type":"application/json" }},
      body: JSON.stringify({{
        prefix: prefix,
        rows: rows,
        cols: cols,
        sync_array: syncArray,
        invert: invertToggle.checked ? true : false  // ✅ NEW
      }})
    }})
    .then(rsp => rsp.json())
    .then(data => {{
      if (data.status === "ok") {{
        infoDiv.textContent = "已保存到 checkedcases: " + data.msg;
      }} else {{
        infoDiv.textContent = "保存失败: " + (data.msg || "");
      }}
    }})
    .catch(err => {{
      console.error(err);
      infoDiv.textContent = "保存出错";
    }});
  }});

  // ✅ NEW: 反转开关变化就重载当前 case
  invertToggle.addEventListener("change", () => {{
    loadCase(currentIndex);
  }});

  fetchCases();
</script>
</body>
</html>
    """
    return html


@app.route("/api/cases")
def api_cases():
    cases = list_cases()
    payload = []
    for i, c in enumerate(cases):
        payload.append(
            {
                "index": i,
                "prefix": c["prefix"],
                "case_id": parse_case_id(c["prefix"]),
                "has_sync": bool(c["has_sync"]),
            }
        )
    return jsonify({"cases": payload})


@app.route("/api/case/<int:index>")
def api_case(index: int):
    invert = parse_bool_param(request.args.get("invert"))  # ✅ NEW

    cases = list_cases()
    if index < 0 or index >= len(cases):
        return jsonify({"error": "index out of range"}), 400

    c = cases[index]
    prefix = c["prefix"]

    # ORI
    ori_path = c.get("show_ori_path")
    ori_img = None
    ori_w = ori_h = None
    if ori_path and os.path.exists(ori_path):
        ori_img = cv2.imread(ori_path, cv2.IMREAD_GRAYSCALE)
        if ori_img is not None:
            ori_h, ori_w = ori_img.shape[:2]
    ori_data_url = file_to_data_url(ori_path)

    # binary：优先已有，否则 OTSU+MORPH
    binary_path = c.get("binary_raw_path")
    bin_img = None
    if binary_path and os.path.exists(binary_path):
        bin_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    else:
        bin_img = otsu_and_morph(ori_img)

    # ✅ NEW: invert
    bin_img = invert_binary(bin_img, invert)

    binary_h = binary_w = None
    if bin_img is not None:
        binary_h, binary_w = bin_img.shape[:2]
    binary_data_url = gray_img_to_data_url(bin_img, fmt=".png")

    # sync
    rows = cols = None
    sync_list = None
    if c.get("has_sync") and c.get("sync_path") and os.path.exists(c["sync_path"]):
        sync_arr = np.load(c["sync_path"]).astype(int)
        rows, cols = sync_arr.shape
        sync_list = sync_arr.tolist()

    return jsonify(
        {
            "prefix": prefix,
            "has_sync": bool(c.get("has_sync")),
            "rows": int(rows) if rows is not None else None,
            "cols": int(cols) if cols is not None else None,
            "sync_array": sync_list,

            "ori_data_url": ori_data_url,
            "ori_w": int(ori_w) if ori_w is not None else None,
            "ori_h": int(ori_h) if ori_h is not None else None,

            "binary_data_url": binary_data_url,
            "binary_w": int(binary_w) if binary_w is not None else None,
            "binary_h": int(binary_h) if binary_h is not None else None,

            "invert": bool(invert),  # ✅ NEW: 回传给前端（可选）
        }
    )


@app.route("/api/init_sync", methods=["POST"])
def api_init_sync():
    data = request.get_json(force=True)
    prefix = data.get("prefix")
    rows = data.get("rows")
    cols = data.get("cols")
    if not prefix or not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0:
        return jsonify({"status": "error", "msg": "invalid prefix/rows/cols"}), 400

    sync = np.ones((rows, cols), dtype=np.uint8)
    return jsonify({"status": "ok", "rows": rows, "cols": cols, "sync_array": sync.tolist()})


def find_case_by_prefix(prefix: str):
    for c in list_cases():
        if c["prefix"] == prefix:
            return c
    return None


def copy_if_exists(src: str, dst_dir: str):
    if src and os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        return dst
    return None


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True)
    prefix = data.get("prefix")
    rows = data.get("rows")
    cols = data.get("cols")
    sync_array = data.get("sync_array")
    invert = parse_bool_param(data.get("invert"))  # ✅ NEW

    if not prefix or not isinstance(rows, int) or not isinstance(cols, int) or sync_array is None:
        return jsonify({"status": "error", "msg": "missing fields"}), 400

    sync_arr = np.array(sync_array, dtype=np.uint8)
    if sync_arr.ndim != 2 or sync_arr.shape != (rows, cols):
        return jsonify({"status": "error", "msg": f"sync_array shape mismatch, expect ({rows},{cols})"}), 400

    c = find_case_by_prefix(prefix)
    if c is None:
        return jsonify({"status": "error", "msg": "prefix not found in badcases"}), 400

    os.makedirs(CHECKED_DIR, exist_ok=True)

    # 1) 保存 sync npy
    sync_npy_path = os.path.join(CHECKED_DIR, f"{prefix}_sync_dm_array.npy")
    np.save(sync_npy_path, sync_arr)

    # 2) 保存 sync DM 图（checked）
    sync_img = dm_array_to_image(sync_arr, cell_w=SYNC_CELL_WIDTH, border=BORDER_WIDTH)
    sync_img_path = os.path.join(CHECKED_DIR, f"{prefix}_sync_dm_checked.png")
    cv2.imwrite(sync_img_path, sync_img)

    # 3) 拷贝 ORI（保持后缀一致）
    copied_ori = None
    if c.get("ori_path"):
        copied_ori = copy_if_exists(c["ori_path"], CHECKED_DIR)
    if copied_ori is None and c.get("ori_pad10_path"):
        copied_ori = copy_if_exists(c["ori_pad10_path"], CHECKED_DIR)
    if copied_ori is None and c.get("dm_image_path"):
        copied_ori = copy_if_exists(c["dm_image_path"], CHECKED_DIR)

    # 4) 准备 binary：优先用已有，否则从 ORI 现算
    binary_src_path = c.get("binary_raw_path")
    bin_img = None

    if binary_src_path and os.path.exists(binary_src_path):
        # 读原始 binary
        bin_img = cv2.imread(binary_src_path, cv2.IMREAD_GRAYSCALE)
        if bin_img is None:
            return jsonify({"status": "error", "msg": "failed to read existing binary"}), 500
    else:
        ori_path = c.get("show_ori_path")
        ori_img = cv2.imread(ori_path, cv2.IMREAD_GRAYSCALE) if ori_path and os.path.exists(ori_path) else None
        bin_img = otsu_and_morph(ori_img)
        if bin_img is None:
            return jsonify({"status": "error", "msg": "failed to build binary from ori"}), 500

    # ✅ NEW: 保存前也做 invert（保证展示与落盘一致）
    bin_img = invert_binary(bin_img, invert)

    bin_h, bin_w = bin_img.shape[:2]

    # 5) 保存 binary_raw（统一保存成 checkedcases 风格命名）
    binary_name = f"{prefix}_binary_dm_image_w{bin_w}h{bin_h}.jpg"
    binary_path = os.path.join(CHECKED_DIR, binary_name)
    cv2.imwrite(binary_path, bin_img)

    # 6) 保存 binary_rect（带网格线）
    rect_bgr = draw_grid_overlay_on_binary(bin_img, rows=rows, cols=cols, color=(0, 255, 0), thickness=1)
    rect_name = f"{prefix}_binary_dm_image_rect_w{bin_w}h{bin_h}.jpg"
    rect_path = os.path.join(CHECKED_DIR, rect_name)
    cv2.imwrite(rect_path, rect_bgr)

    msg_parts = [
        f"sync_npy={os.path.basename(sync_npy_path)}",
        f"sync_checked={os.path.basename(sync_img_path)}",
        f"binary_raw={os.path.basename(binary_path)}",
        f"binary_rect={os.path.basename(rect_path)}",
        f"invert={int(invert)}",
    ]
    if copied_ori:
        msg_parts.append(f"ori={os.path.basename(copied_ori)}")

    return jsonify({"status": "ok", "msg": "; ".join(msg_parts)})


if __name__ == "__main__":
    print("Running on http://127.0.0.1:5000  (Ctrl+C 退出)")
    print(f"BADCASES_DIR = {BADCASES_DIR}")
    print(f"CHECKED_DIR  = {CHECKED_DIR}")
    app.run(host="127.0.0.1", port=5000, debug=True)
