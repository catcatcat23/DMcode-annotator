#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
import shutil
import sys
import cv2
import numpy as np
from flask import Flask, jsonify, request

# ================== 配置区 ==================
BADCASES_DIR = "/home/cat/workspace/DMCODE/SNcode/badcases"
base = os.path.dirname(BADCASES_DIR)
CHECKED_DIR = os.path.join(base, "checkedcases")
os.makedirs(CHECKED_DIR, exist_ok=True)

# 前端网格 cell 像素大小
CELL_SIZE_PX = 16

# 生成 DM 图时每个 cell 的宽度、外边白边
SYNC_CELL_WIDTH = 5
BORDER_WIDTH = 10
# ==========================================

app = Flask(__name__)


def list_cases():
    """
    在 BADCASES_DIR 下扫描所有 *_sync_dm_array.npy，
    每一个对应一个 prefix。
    """
    cases = []
    for fname in sorted(os.listdir(BADCASES_DIR)):
        if not fname.endswith("_sync_dm_array.npy"):
            continue
        prefix = fname[: -len("_sync_dm_array.npy")]

        # 原始 DM 裁剪图
        dm_img_path = None
        for f2 in os.listdir(BADCASES_DIR):
            if f2.startswith(prefix + "_dm_image_"):
                dm_img_path = os.path.join(BADCASES_DIR, f2)
                break

        # 原始二值图 & 带网格二值图
        binary_raw_img_path = None
        binary_rect_img_path = None
        for f2 in os.listdir(BADCASES_DIR):
            if f2.startswith(prefix + "_binary_dm_image_rect_"):
                binary_rect_img_path = os.path.join(BADCASES_DIR, f2)
            elif f2.startswith(prefix + "_binary_dm_image_"):
                # 注意：这里用 elif，避免 rect 版本再次被当成 raw
                binary_raw_img_path = os.path.join(BADCASES_DIR, f2)

        sync_npy = os.path.join(BADCASES_DIR, fname)
        encoder_npy = os.path.join(BADCASES_DIR, prefix + "_encoder_dm_array.npy")
        has_encoder = os.path.exists(encoder_npy)

        cases.append(
            {
                "prefix": prefix,
                "dm_image_path": dm_img_path,
                "binary_raw_image_path": binary_raw_img_path,
                "binary_rect_image_path": binary_rect_img_path,
                "sync_array_path": sync_npy,
                "encoder_array_path": encoder_npy if has_encoder else None,
                "has_encoder": has_encoder,
            }
        )

    return cases


def image_to_data_url(path):
    """把本地图片文件转成 data URL，方便前端 <img src=...> 直接用。"""
    if path is None or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def dm_array_to_image(dm_array, cell_w=SYNC_CELL_WIDTH, border=BORDER_WIDTH):
    """
    把 0/1 的 dm_array 还原成一个黑白 DM 图（1=白，0=黑），带白边。
    """
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


import base64  # 别忘了引入


@app.route("/")
def index():
    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8" />
    <title>DM Code Cell 可视化编辑器</title>
    <style>
        :root {{
            --bg: #050916;
            --panel-bg: #0b1020;
            --accent: #4fd1c5;
            --text-main: #e5edf7;
            --text-sub: #9aa4bf;
            --highlight: #ffd93b; /* 黄色高亮 */
        }}

        * {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            padding: 24px 32px 40px;
            background: radial-gradient(circle at top, #101a33 0, #050916 52%, #02040a 100%);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        }}

        #app {{
            max-width: 1280px;
            margin: 0 auto;
        }}

        #controls {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }}

        .pill-btn {{
            padding: 6px 14px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.08);
            background: radial-gradient(circle at top left, rgba(79,209,197,0.28), rgba(11,16,32,0.95));
            color: #f3f7ff;
            cursor: pointer;
            font-size: 13px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .pill-btn.secondary {{
            background: rgba(11,16,32,0.9);
            border-color: rgba(255,255,255,0.08);
            color: var(--text-sub);
        }}

        .pill-btn:hover {{ border-color: var(--accent); }}

        #indexInfo {{
            font-size: 13px;
            color: var(--text-sub);
        }}

        #info {{
            font-size: 13px;
            color: var(--accent);
            min-height: 18px;
            margin-bottom: 10px;
        }}

        .section-title {{
            font-size: 11px;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.28em;
            margin: 12px 4px 6px;
        }}

        .panel {{
            background: radial-gradient(circle at top left, rgba(76,141,245,0.08), var(--panel-bg));
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.04);
            box-shadow: 0 20px 40px rgba(0,0,0,0.65);
            padding: 14px 16px 16px;
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}

        .panel-title-main {{ font-size: 13px; }}
        .panel-sub {{ font-size: 11px; color: var(--text-sub); margin-top: 4px; }}

        .pill-tag {{
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(8, 16, 38, 0.9);
            border: 1px solid rgba(255,255,255,0.1);
            font-size: 10px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-sub);
        }}

        .toggle-wrap {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
            color: var(--text-sub);
        }}

        .toggle-wrap input {{
            accent-color: var(--accent);
        }}

        .layout-middle {{
            display: flex;
            justify-content: center;
            margin-bottom: 22px;
        }}

        .layout-bottom {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .canvas-wrapper {{
            display: flex;
            justify-content: center;
            align-items: center;
            padding-top: 6px;
        }}

        canvas {{
            border-radius: 16px;
            background: #050814;
            border: 1px solid rgba(255,255,255,0.06);
        }}

        .small-input {{
            width: 72px;
            background: transparent;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.16);
            color: var(--text-main);
            padding: 2px 6px;
            font-size: 12px;
            outline: none;
        }}

        .small-input::placeholder {{
            color: rgba(154,164,191,0.7);
        }}

        .small-select {{
            background: rgba(8,16,38,0.9);
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.16);
            color: var(--text-main);
            font-size: 12px;
            padding: 2px 6px;
            outline: none;
        }}
    </style>
</head>
<body>
<div id="app">
    <div id="controls">
        <button id="prevBtn" class="pill-btn secondary">Prev</button>
        <button id="nextBtn" class="pill-btn secondary">Next</button>
        <button id="saveBtn" class="pill-btn">Save to checkedcases</button>
        <span id="indexInfo"></span>

        <!-- ✅ 修改点：默认只看“可识别（有 encoder）” -->
        <label class="toggle-wrap">
            <input type="checkbox" id="encoderOnly" checked />
            <span>只看可识别（有 encoder）</span>
        </label>

        <label class="toggle-wrap">
            <span>ID 范围:</span>
            <input type="number" id="minIdInput" class="small-input" placeholder="min" />
            <span>-</span>
            <input type="number" id="maxIdInput" class="small-input" placeholder="max" />
            <button id="applyFilterBtn" class="pill-btn secondary" style="padding:2px 8px;font-size:11px;">应用</button>
        </label>

        <label class="toggle-wrap">
            <span>排序:</span>
            <select id="sortMode" class="small-select">
                <option value="original">原始</option>
                <option value="id_asc">ID↑</option>
                <option value="id_desc">ID↓</option>
            </select>
        </label>
    </div>

    <div id="info"></div>

    <div class="section-title">SYNC CELLS</div>
    <div class="layout-middle">
        <div class="panel" style="width: 360px;">
            <div class="panel-header">
                <div>
                    <div class="panel-title-main">Sync Cells</div>
                    <div class="panel-sub">
                        点击修改 sync 矩阵（1=白, 0=黑） ·
                        <span id="syncSizeSpan"></span>
                    </div>
                </div>
                <div class="pill-tag">SYNC · 可点击</div>
            </div>
            <div class="canvas-wrapper">
                <canvas id="syncCanvas"></canvas>
            </div>
        </div>
    </div>

    <div class="section-title">BINARY · ENCODER 对比</div>
    <div class="layout-bottom">
        <div class="panel">
            <div class="panel-header">
                <div>
                    <div class="panel-title-main">二值图（带网格 / 高亮）</div>
                    <div class="panel-sub">
                        基于 binary_dm_image_rect + 高亮 ·
                        <span id="binarySizeSpan"></span>
                    </div>
                </div>
                <div class="pill-tag">BINARY</div>
            </div>
            <div class="canvas-wrapper">
                <canvas id="binaryCanvas"></canvas>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">
                <div>
                    <div class="panel-title-main">Encoder Cells（可点击）</div>
                    <div class="panel-sub">
                        点击 encoder 某 cell：翻转 0/1，并在左侧同步高亮 ·
                        <span id="encoderSizeSpan"></span>
                    </div>
                </div>
                <div class="pill-tag">ENCODER</div>
            </div>
            <div class="canvas-wrapper">
                <canvas id="encoderCanvas"></canvas>
            </div>
        </div>
    </div>
</div>

<script>
    const CELL_SIZE = {CELL_SIZE_PX};
    const HIGHLIGHT_COLOR = "#ffd93b";

    let allCases = [];
    let cases = [];
    let currentIndex = 0;

    let rows = 0;
    let cols = 0;
    let syncArray = [];
    let encoderArray = [];
    let hasEncoder = false;

    let highlightRow = null;
    let highlightCol = null;

    let binaryImgDataUrl = null;

    const syncCanvas = document.getElementById("syncCanvas");
    const encoderCanvas = document.getElementById("encoderCanvas");
    const binaryCanvas = document.getElementById("binaryCanvas");
    const syncCtx = syncCanvas.getContext("2d");
    const encoderCtx = encoderCanvas.getContext("2d");
    const binaryCtx = binaryCanvas.getContext("2d");

    const indexInfo = document.getElementById("indexInfo");
    const infoDiv = document.getElementById("info");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const saveBtn = document.getElementById("saveBtn");

    // ✅ 修改点：encoderOnly
    const encoderOnly = document.getElementById("encoderOnly");

    const syncSizeSpan = document.getElementById("syncSizeSpan");
    const binarySizeSpan = document.getElementById("binarySizeSpan");
    const encoderSizeSpan = document.getElementById("encoderSizeSpan");

    const minIdInput = document.getElementById("minIdInput");
    const maxIdInput = document.getElementById("maxIdInput");
    const applyFilterBtn = document.getElementById("applyFilterBtn");
    const sortModeSelect = document.getElementById("sortMode");

    function fetchCases() {{
        fetch("/api/cases")
            .then(r => r.json())
            .then(data => {{
                allCases = data.cases || [];
                applyCaseFilter();
            }})
            .catch(err => {{
                console.error(err);
                infoDiv.textContent = "加载 cases 时出错";
            }});
    }}

    function applyCaseFilter() {{
        if (!allCases.length) {{
            infoDiv.textContent = "在 badcases 目录下没有找到 *_sync_dm_array.npy";
            cases = [];
            indexInfo.textContent = "";
            return;
        }}

        let filtered = allCases.slice();

        // ✅ 修改点：只看可识别（has_encoder == true）
        if (encoderOnly.checked) {{
            filtered = filtered.filter(c => c.has_encoder);
        }}

        let minId = parseInt(minIdInput.value);
        let maxId = parseInt(maxIdInput.value);

        if (!Number.isNaN(minId)) {{
            filtered = filtered.filter(c => c.case_id != null && c.case_id >= minId);
        }}
        if (!Number.isNaN(maxId)) {{
            filtered = filtered.filter(c => c.case_id != null && c.case_id <= maxId);
        }}

        const mode = sortModeSelect.value;
        if (mode === "id_asc") {{
            filtered.sort((a, b) => {{
                const ida = (a.case_id != null ? a.case_id : Number.MAX_SAFE_INTEGER);
                const idb = (b.case_id != null ? b.case_id : Number.MAX_SAFE_INTEGER);
                return ida - idb;
            }});
        }} else if (mode === "id_desc") {{
            filtered.sort((a, b) => {{
                const ida = (a.case_id != null ? a.case_id : -1);
                const idb = (b.case_id != null ? b.case_id : -1);
                return idb - ida;
            }});
        }}

        if (!filtered.length) {{
            cases = [];
            indexInfo.textContent = "";
            infoDiv.textContent = "当前筛选条件下没有样本";
            syncCtx.clearRect(0, 0, syncCanvas.width, syncCanvas.height);
            encoderCtx.clearRect(0, 0, encoderCanvas.width, encoderCanvas.height);
            binaryCtx.clearRect(0, 0, binaryCanvas.width, binaryCanvas.height);
            return;
        }}

        cases = filtered;
        currentIndex = 0;
        infoDiv.textContent = "";
        loadCase(currentIndex);
    }}

    function loadCase(idx) {{
        if (!cases.length) return;
        if (idx < 0 || idx >= cases.length) return;
        currentIndex = idx;
        const c = cases[currentIndex];
        const serverIndex = c.index;

        const idText = (c.case_id != null) ? ` · ID: ${{c.case_id}}` : "";
        indexInfo.textContent = `当前第 ${{currentIndex+1}} / ${{cases.length}} 个 · prefix: ${{c.prefix}}${{idText}}`;
        infoDiv.textContent = "";

        fetch(`/api/case/${{serverIndex}}`)
            .then(r => r.json())
            .then(data => {{
                rows = data.rows;
                cols = data.cols;
                syncArray = data.sync_array;
                encoderArray = data.encoder_array;
                hasEncoder = data.has_encoder;
                binaryImgDataUrl = data.binary_dm_data_url;

                syncSizeSpan.textContent = `${{data.rows}} × ${{data.cols}}`;

                if (data.encoder_rows && data.encoder_cols) {{
                    encoderSizeSpan.textContent = `${{data.encoder_rows}} × ${{data.encoder_cols}}`;
                }} else {{
                    encoderSizeSpan.textContent = `${{data.rows}} × ${{data.cols}}`;
                }}

                if (data.binary_h && data.binary_w) {{
                    binarySizeSpan.textContent =
                        `${{data.binary_h}} × ${{data.binary_w}}px · grid ${{data.rows}} × ${{data.cols}}`;
                }} else {{
                    binarySizeSpan.textContent = `grid ${{data.rows}} × ${{data.cols}}`;
                }}

                highlightRow = null;
                highlightCol = null;

                drawSyncCanvas();
                drawEncoderCanvas();
                drawBinaryCanvas();
            }})
            .catch(err => {{
                console.error(err);
                infoDiv.textContent = "加载 case 失败";
            }});
    }}

    function drawGrid(ctx, rows, cols, cellSize, color, alpha=0.4) {{
        ctx.save();
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = 0.5;

        for (let i = 0; i <= rows; i++) {{
            const y = i * cellSize + 0.5;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(cols * cellSize, y);
            ctx.stroke();
        }}
        for (let j = 0; j <= cols; j++) {{
            const x = j * cellSize + 0.5;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, rows * cellSize);
            ctx.stroke();
        }}
        ctx.restore();
    }}

    function drawArray(ctx, arr, rows, cols, cellSize, highlightR=null, highlightC=null) {{
        ctx.clearRect(0, 0, cols * cellSize, rows * cellSize);

        for (let i = 0; i < rows; i++) {{
            for (let j = 0; j < cols; j++) {{
                const v = arr[i][j];
                ctx.fillStyle = v ? "#ffffff" : "#050816";
                ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }}
        }}
        drawGrid(ctx, rows, cols, cellSize, "rgba(255,255,255,0.35)", 0.4);

        if (highlightR !== null && highlightC !== null) {{
            ctx.save();
            ctx.strokeStyle = HIGHLIGHT_COLOR;
            ctx.lineWidth = 2;
            ctx.shadowColor = HIGHLIGHT_COLOR;
            ctx.shadowBlur = 12;
            const x = highlightC * cellSize + 1;
            const y = highlightR * cellSize + 1;
            ctx.strokeRect(x, y, cellSize - 2, cellSize - 2);
            ctx.restore();
        }}
    }}

    function drawSyncCanvas() {{
        syncCanvas.width = cols * CELL_SIZE;
        syncCanvas.height = rows * CELL_SIZE;
        drawArray(syncCtx, syncArray, rows, cols, CELL_SIZE, highlightRow, highlightCol);
    }}

    function drawEncoderCanvas() {{
        encoderCanvas.width = cols * CELL_SIZE;
        encoderCanvas.height = rows * CELL_SIZE;
        drawArray(encoderCtx, encoderArray, rows, cols, CELL_SIZE, highlightRow, highlightCol);
    }}

    function drawBinaryCanvas() {{
        binaryCanvas.width = cols * CELL_SIZE;
        binaryCanvas.height = rows * CELL_SIZE;
        const w = cols * CELL_SIZE;
        const h = rows * CELL_SIZE;

        binaryCtx.save();
        binaryCtx.fillStyle = "rgba(0,0,0,0.9)";
        binaryCtx.fillRect(0, 0, w, h);

        if (binaryImgDataUrl) {{
            const img = new Image();
            img.onload = () => {{
                binaryCtx.save();
                binaryCtx.globalAlpha = 0.9;
                binaryCtx.drawImage(img, 0, 0, w, h);
                binaryCtx.restore();

                drawGrid(binaryCtx, rows, cols, CELL_SIZE, "rgba(0,255,128,0.75)", 0.7);

                if (highlightRow !== null && highlightCol !== null) {{
                    binaryCtx.save();
                    binaryCtx.strokeStyle = HIGHLIGHT_COLOR;
                    binaryCtx.lineWidth = 2;
                    binaryCtx.shadowColor = HIGHLIGHT_COLOR;
                    binaryCtx.shadowBlur = 14;
                    const x = highlightCol * CELL_SIZE + 1;
                    const y = highlightRow * CELL_SIZE + 1;
                    binaryCtx.strokeRect(x, y, CELL_SIZE - 2, CELL_SIZE - 2);
                    binaryCtx.restore();
                }}
            }};
            img.src = binaryImgDataUrl;
        }} else {{
            drawGrid(binaryCtx, rows, cols, CELL_SIZE, "rgba(0,255,128,0.45)", 0.6);
        }}

        binaryCtx.restore();
    }}

    function toggleCell(arr, x, y) {{
        const j = Math.floor(x / CELL_SIZE);
        const i = Math.floor(y / CELL_SIZE);
        if (i < 0 || i >= rows || j < 0 || j >= cols) return null;
        arr[i][j] = arr[i][j] ? 0 : 1;
        return {{ i, j }};
    }}

    function updateHighlight(i, j) {{
        highlightRow = i;
        highlightCol = j;
        drawSyncCanvas();
        drawEncoderCanvas();
        drawBinaryCanvas();
    }}

    syncCanvas.addEventListener("click", (e) => {{
        const rect = syncCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const idx = toggleCell(syncArray, x, y);
        if (idx) {{
            updateHighlight(idx.i, idx.j);
        }}
    }});

    encoderCanvas.addEventListener("click", (e) => {{
        const rect = encoderCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const idx = toggleCell(encoderArray, x, y);
        if (idx) {{
            updateHighlight(idx.i, idx.j);
        }}
    }});

    prevBtn.addEventListener("click", () => {{
        if (!cases.length) return;
        const idx = (currentIndex - 1 + cases.length) % cases.length;
        loadCase(idx);
    }});

    nextBtn.addEventListener("click", () => {{
        if (!cases.length) return;
        const idx = (currentIndex + 1) % cases.length;
        loadCase(idx);
    }});

    saveBtn.addEventListener("click", () => {{
        if (!cases.length) return;
        const c = cases[currentIndex];

        fetch("/api/save", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{
                prefix: c.prefix,
                sync_array: syncArray,
                encoder_array: encoderArray
            }})
        }})
        .then(r => r.json())
        .then(data => {{
            if (data.status === "ok") {{
                infoDiv.textContent = "已保存到 checkedcases: " + data.msg;
            }} else {{
                infoDiv.textContent = "保存失败: " + data.msg;
            }}
        }})
        .catch(err => {{
            console.error(err);
            infoDiv.textContent = "保存时出错";
        }});
    }});

    // ✅ 修改点：监听 encoderOnly
    encoderOnly.addEventListener("change", applyCaseFilter);

    applyFilterBtn.addEventListener("click", applyCaseFilter);
    sortModeSelect.addEventListener("change", applyCaseFilter);

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
        prefix = c["prefix"]
        case_id = None
        # 从 prefix 开头解析整数 ID，例如 "14_282193_..." -> 14
        try:
            case_id = int(prefix.split("_", 1)[0])
        except Exception:
            case_id = None

        payload.append(
            {
                "index": i,  # 后端固定 index
                "prefix": prefix,
                "has_encoder": c["has_encoder"],
                "case_id": case_id,
            }
        )

    return jsonify({"cases": payload})


@app.route("/api/case/<int:index>")
def api_case(index: int):
    cases = list_cases()
    if index < 0 or index >= len(cases):
        return jsonify({"error": "index out of range"}), 400

    c = cases[index]
    sync_arr = np.load(c["sync_array_path"]).astype(int)
    rows, cols = sync_arr.shape

    if c["encoder_array_path"] is not None and os.path.exists(c["encoder_array_path"]):
        encoder_arr = np.load(c["encoder_array_path"]).astype(int)
    else:
        encoder_arr = np.copy(sync_arr)

    enc_rows, enc_cols = encoder_arr.shape

    # 用 rect 版本做展示，同时取一下像素宽高
    binary_dm_data_url = image_to_data_url(c["binary_rect_image_path"])
    binary_h = None
    binary_w = None
    if c["binary_rect_image_path"] is not None and os.path.exists(c["binary_rect_image_path"]):
        img = cv2.imread(c["binary_rect_image_path"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            binary_h, binary_w = img.shape[:2]

    return jsonify(
        {
            "prefix": c["prefix"],
            "rows": int(rows),
            "cols": int(cols),
            "sync_array": sync_arr.tolist(),
            "encoder_array": encoder_arr.tolist(),
            "has_encoder": c["encoder_array_path"] is not None,
            "binary_dm_data_url": binary_dm_data_url,
            "encoder_rows": int(enc_rows),
            "encoder_cols": int(enc_cols),
            "binary_h": int(binary_h) if binary_h is not None else None,
            "binary_w": int(binary_w) if binary_w is not None else None,
        }
    )


def copy_binary_images(prefix: str):
    """
    拷贝到 CHECKED_DIR：
      - prefix_binary_dm_image_*.jpg         （原始二值图）
      - prefix_binary_dm_image_rect_*.jpg   （带网格二值图）
      - prefix_sync_dm_code_*.jpg           （原始 sync cell 图）
    返回 (raw_dst, rect_dst, sync_dst)
    """
    raw_dst = None
    rect_dst = None
    sync_dst = None

    for f in os.listdir(BADCASES_DIR):
        if f.startswith(prefix + "_binary_dm_image_rect_"):
            src = os.path.join(BADCASES_DIR, f)
            rect_dst = os.path.join(CHECKED_DIR, f)
            shutil.copy2(src, rect_dst)

        elif f.startswith(prefix + "_binary_dm_image_"):
            src = os.path.join(BADCASES_DIR, f)
            raw_dst = os.path.join(CHECKED_DIR, f)
            shutil.copy2(src, raw_dst)

        elif f.startswith(prefix + "_sync_dm_code_"):
            src = os.path.join(BADCASES_DIR, f)
            sync_dst = os.path.join(CHECKED_DIR, f)
            shutil.copy2(src, sync_dst)

    return raw_dst, rect_dst, sync_dst


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True)
    prefix = data.get("prefix")
    sync_array = data.get("sync_array")
    encoder_array = data.get("encoder_array")

    if prefix is None or sync_array is None or encoder_array is None:
        return jsonify({"status": "error", "msg": "missing fields"}), 400

    sync_arr = np.array(sync_array, dtype=np.uint8)
    enc_arr = np.array(encoder_array, dtype=np.uint8)

    os.makedirs(CHECKED_DIR, exist_ok=True)

    # 修改后的 sync / encoder npy
    sync_npy_path = os.path.join(CHECKED_DIR, f"{prefix}_sync_dm_array.npy")
    enc_npy_path = os.path.join(CHECKED_DIR, f"{prefix}_encoder_dm_array.npy")
    np.save(sync_npy_path, sync_arr)
    np.save(enc_npy_path, enc_arr)

    # 修改后的 DM 图
    sync_img = dm_array_to_image(sync_arr, cell_w=SYNC_CELL_WIDTH, border=BORDER_WIDTH)
    sync_img_path = os.path.join(CHECKED_DIR, f"{prefix}_sync_dm_checked.png")
    cv2.imwrite(sync_img_path, sync_img)

    enc_img = dm_array_to_image(enc_arr, cell_w=SYNC_CELL_WIDTH, border=BORDER_WIDTH)
    enc_img_path = os.path.join(CHECKED_DIR, f"{prefix}_encoder_dm_checked.png")
    cv2.imwrite(enc_img_path, enc_img)

    # 拷贝原始二值图 + 带网格二值图 + 原始 sync
    raw_binary_dst, rect_binary_dst, sync_dst = copy_binary_images(prefix)

    msg_parts = [
        f"sync_npy={os.path.basename(sync_npy_path)}",
        f"encoder_npy={os.path.basename(enc_npy_path)}",
        f"sync_dm_jpg={os.path.basename(sync_img_path)}",
        f"encoder_dm_jpg={os.path.basename(enc_img_path)}",
    ]
    if raw_binary_dst:
        msg_parts.append(f"binary_raw={os.path.basename(raw_binary_dst)}")
    if rect_binary_dst:
        msg_parts.append(f"binary_rect={os.path.basename(rect_binary_dst)}")
    if sync_dst:
        msg_parts.append(f"sync_dm_code={os.path.basename(sync_dst)}")

    return jsonify({"status": "ok", "msg": "; ".join(msg_parts)})


if __name__ == "__main__":
    print("Running on http://127.0.0.1:5000  (Ctrl+C 退出)")
    app.run(host="127.0.0.1", port=5000, debug=True)
