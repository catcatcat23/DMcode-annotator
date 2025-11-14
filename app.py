import os
import base64
import io

import numpy as np
import cv2
from flask import Flask, jsonify, request

# ================== 配置区 ==================
# 你的 badcases 目录
BADCASES_DIR = "/home/cat/workspace/DMCODE/SNcode/badcases"
base = os.path.dirname(BADCASES_DIR)
CHECKED_DIR = os.path.join(base, "checkedcases")
os.makedirs(CHECKED_DIR, exist_ok=True)

# 前端网格的 cell 像素大小
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

        # 找对应的裁剪后 dm 原图
        dm_img_path = None
        for f2 in os.listdir(BADCASES_DIR):
            if f2.startswith(prefix + "_dm_image_"):
                dm_img_path = os.path.join(BADCASES_DIR, f2)
                break

        # 二值化图（带网格的原 pipeline binary_dm_image_show）
        binary_img_path = None
        for f2 in os.listdir(BADCASES_DIR):
            if f2.startswith(prefix + "_binary_dm_image_rect_"):
                binary_img_path = os.path.join(BADCASES_DIR, f2)
                break

        sync_npy = os.path.join(BADCASES_DIR, fname)
        encoder_npy = os.path.join(BADCASES_DIR, prefix + "_encoder_dm_array.npy")
        has_encoder = os.path.exists(encoder_npy)

        cases.append(
            {
                "prefix": prefix,
                "dm_image_path": dm_img_path,
                "binary_image_path": binary_img_path,
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
    # 简单按 jpg 处理
    return f"data:image/jpeg;base64,{b64}"


def dm_array_to_image(dm_array, cell_w=SYNC_CELL_WIDTH, border=BORDER_WIDTH):
    """
    把 0/1 的 dm_array 还原成一个黑白 DM 图（1=白，0=黑），带白边。
    """
    rows, cols = dm_array.shape
    h = rows * cell_w + 2 * border
    w = cols * cell_w + 2 * border
    img = np.ones((h, w), dtype=np.uint8) * 255  # 全白背景

    for i in range(rows):
        for j in range(cols):
            if dm_array[i, j] == 0:  # 0=黑
                y1 = border + i * cell_w
                y2 = border + (i + 1) * cell_w
                x1 = border + j * cell_w
                x2 = border + (j + 1) * cell_w
                img[y1:y2, x1:x2] = 0

    return img


@app.route("/")
def index():
    """
    返回一个带前端的单页应用（HTML+JS 全在这里）。
    """
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
            --panel-border: #283046;
            --accent: #4fd1c5;
            --accent-soft: rgba(79, 209, 197, 0.16);
            --text-main: #e5edf7;
            --text-sub: #9aa4bf;
            --grid-line: rgba(255, 255, 255, 0.08);
            --binary-grid: rgba(46, 255, 144, 0.45);
            --binary-bg-tint: rgba(0, 255, 128, 0.16);
            --highlight: #00ffd0;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 24px 32px 40px;
            background: radial-gradient(circle at top, #101a33 0, #050916 52%, #02040a 100%);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
        }}

        h2, h3 {{
            margin: 0;
            font-weight: 500;
            letter-spacing: 0.04em;
        }}

        #app {{
            max-width: 1280px;
            margin: 0 auto;
        }}

        #controls {{
            display: flex;
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

        .pill-btn:hover {{
            border-color: var(--accent);
        }}

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

        .badge {{
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            font-size: 10px;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--text-sub);
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
            box-shadow:
                0 20px 40px rgba(0,0,0,0.65),
                0 0 0 1px rgba(81, 92, 141, 0.25) inset;
            padding: 14px 16px 16px;
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}

        .panel-title-main {{
            font-size: 13px;
        }}

        .panel-sub {{
            font-size: 11px;
            color: var(--text-sub);
            margin-top: 4px;
        }}

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

        .layout-top {{
            display: flex;
            justify-content: center;
            margin-bottom: 18px;
        }}

        #origImg {{
            max-width: 200px;
            max-height: 200px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.16);
            background: #000;
            object-fit: contain;
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

        /* Binary canvas 上做绿色网格/高亮的效果，靠绘制实现 */
    </style>
</head>
<body>
<div id="app">
    <div id="controls">
        <button id="prevBtn" class="pill-btn secondary">Prev</button>
        <button id="nextBtn" class="pill-btn secondary">Next</button>
        <button id="saveBtn" class="pill-btn">Save to checkedcases</button>
        <span id="indexInfo"></span>
    </div>
    <div id="info"></div>

    <!-- 顶部：原始 DM 图 -->
    <div class="section-title">原图 (DM_IMAGE)</div>
    <div class="layout-top">
        <div class="panel" style="width: 260px;">
            <div class="panel-header">
                <div>
                    <div class="panel-title-main">原始 DM 裁剪图</div>
                    <div class="panel-sub">来自 pipeline 的 {{"dm_image"}}</div>
                </div>
                <div class="pill-tag">Reference</div>
            </div>
            <div class="canvas-wrapper">
                <img id="origImg" src="" alt="no image" />
            </div>
        </div>
    </div>

    <!-- 中部：Sync Cells -->
    <div class="section-title">SYNC CELLS</div>
    <div class="layout-middle">
        <div class="panel" style="width: 360px;">
            <div class="panel-header">
                <div>
                    <div class="panel-title-main">Sync Cells</div>
                    <div class="panel-sub">点击修改 sync 矩阵（1=白, 0=黑）</div>
                </div>
                <div class="pill-tag">Sync · 可点击</div>
            </div>
            <div class="canvas-wrapper">
                <canvas id="syncCanvas"></canvas>
            </div>
        </div>
    </div>

    <!-- 底部：左 Binary，右 Encoder -->
    <div class="section-title">BINARY · ENCODER 对比</div>
    <div class="layout-bottom">
        <!-- 左：binary -->
        <div class="panel">
            <div class="panel-header">
                <div>
                    <div class="panel-title-main">二值图（带网格 / 高亮）</div>
                    <div class="panel-sub">左：pipeline 的 binary_dm_image_show，经增强 + 网格 + 高亮</div>
                </div>
                <div class="pill-tag">Binary</div>
            </div>
            <div class="canvas-wrapper">
                <canvas id="binaryCanvas"></canvas>
            </div>
        </div>

        <!-- 右：encoder -->
        <div class="panel">
            <div class="panel-header">
                <div>
                    <div class="panel-title-main">Encoder Cells（可点击）</div>
                    <div class="panel-sub">点击 encoder 某 cell：翻转 0/1，并在左侧同步高亮该 cell</div>
                </div>
                <div class="pill-tag">Encoder</div>
            </div>
            <div class="canvas-wrapper">
                <canvas id="encoderCanvas"></canvas>
            </div>
        </div>
    </div>
</div>

<script>
    const CELL_SIZE = {CELL_SIZE_PX};

    let cases = [];
    let currentIndex = 0;

    // 当前矩阵
    let rows = 0;
    let cols = 0;
    let syncArray = [];
    let encoderArray = [];
    let hasEncoder = false;

    // 当前高亮 cell
    let highlightRow = null;
    let highlightCol = null;

    // Binary 原图 dataURL
    let binaryImgDataUrl = null;

    const origImg = document.getElementById("origImg");
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

    function fetchCases() {{
        fetch("/api/cases")
            .then(r => r.json())
            .then(data => {{
                cases = data.cases;
                if (!cases.length) {{
                    infoDiv.textContent = "在 badcases 目录下没有找到 *_sync_dm_array.npy";
                    return;
                }}
                loadCase(0);
            }})
            .catch(err => {{
                console.error(err);
                infoDiv.textContent = "加载 cases 时出错，请检查后台日志。";
            }});
    }}

    function loadCase(idx) {{
        if (idx < 0 || idx >= cases.length) return;
        currentIndex = idx;
        const c = cases[currentIndex];
        infoDiv.textContent = "";
        indexInfo.textContent = `当前第 ${{currentIndex+1}} / ${{cases.length}} 个 · prefix: ${{c.prefix}}`;

        fetch(`/api/case/${{currentIndex}}`)
            .then(r => r.json())
            .then(data => {{
                rows = data.rows;
                cols = data.cols;
                syncArray = data.sync_array;
                encoderArray = data.encoder_array;
                hasEncoder = data.has_encoder;
                binaryImgDataUrl = data.binary_dm_data_url;

                highlightRow = null;
                highlightCol = null;

                // 原图
                if (data.dm_image_data_url) {{
                    origImg.src = data.dm_image_data_url;
                    origImg.style.display = "block";
                }} else {{
                    origImg.src = "";
                    origImg.style.display = "none";
                }}

                drawSyncCanvas();
                drawEncoderCanvas();
                drawBinaryCanvas();
            }})
            .catch(err => {{
                console.error(err);
                infoDiv.textContent = "加载某个 case 时出错，请检查后台日志。";
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
                // 1 = 白, 0 = 黑
                ctx.fillStyle = v ? "#ffffff" : "#050816";
                ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }}
        }}

        // 网格
        drawGrid(ctx, rows, cols, cellSize, "rgba(255,255,255,0.35)", 0.4);

        // 高亮
        if (highlightR !== null && highlightC !== null) {{
            ctx.save();
            ctx.strokeStyle = "#00ffd0";
            ctx.lineWidth = 2;
            ctx.shadowColor = "#00ffd0";
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
        drawArray(syncCtx, syncArray, rows, cols, CELL_SIZE, null, null);
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
                // 先画一层略带绿色的背景，增强亮度
                binaryCtx.fillStyle = "rgba(0,255,128,0.12)";
                binaryCtx.fillRect(0, 0, w, h);

                // 把原始二值化图拉伸铺满
                binaryCtx.globalAlpha = 0.9;
                binaryCtx.drawImage(img, 0, 0, w, h);
                binaryCtx.restore();

                // 绿色网格
                drawGrid(binaryCtx, rows, cols, CELL_SIZE, "rgba(0,255,128,0.75)", 0.7);

                // 如果有高亮 cell，就在这里画
                if (highlightRow !== null && highlightCol !== null) {{
                    binaryCtx.save();
                    binaryCtx.strokeStyle = "#00ffd0";
                    binaryCtx.lineWidth = 2;
                    binaryCtx.shadowColor = "#00ffd0";
                    binaryCtx.shadowBlur = 14;
                    const x = highlightCol * CELL_SIZE + 1;
                    const y = highlightRow * CELL_SIZE + 1;
                    binaryCtx.strokeRect(x, y, CELL_SIZE - 2, CELL_SIZE - 2);
                    binaryCtx.restore();
                }}
            }};
            img.src = binaryImgDataUrl;
        }} else {{
            // 没有图时，只画网格
            drawGrid(binaryCtx, rows, cols, CELL_SIZE, "rgba(0,255,128,0.45)", 0.6);
        }}

        binaryCtx.restore();
    }}

    function toggleCell(arr, x, y) {{
        const j = Math.floor(x / CELL_SIZE);
        const i = Math.floor(y / CELL_SIZE);
        if (i < 0 || i >= rows || j < 0 || j >= cols) return null;
        arr[i][j] = arr[i][j] ? 0 : 1;
        return {{i, j}};
    }}

    // Sync canvas 点击：只改 syncArray，不联动 binary 高亮
    syncCanvas.addEventListener("click", (e) => {{
        const rect = syncCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const idx = toggleCell(syncArray, x, y);
        if (idx) {{
            drawSyncCanvas();
        }}
    }});

    // Encoder canvas 点击：改 encoderArray + 更新两个高亮
    encoderCanvas.addEventListener("click", (e) => {{
        const rect = encoderCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const idx = toggleCell(encoderArray, x, y);
        if (idx) {{
            highlightRow = idx.i;
            highlightCol = idx.j;
            drawEncoderCanvas();
            drawBinaryCanvas();
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
            headers: {{
                "Content-Type": "application/json"
            }},
            body: JSON.stringify({{
                prefix: c.prefix,
                sync_array: syncArray,
                encoder_array: encoderArray
            }})
        }})
        .then(r => r.json())
        .then(data => {{
            if (data.status === "ok") {{
                infoDiv.textContent = "已保存到 checkedcases 下：" + data.msg;
            }} else {{
                infoDiv.textContent = "保存失败: " + data.msg;
            }}
        }})
        .catch(err => {{
            console.error(err);
            infoDiv.textContent = "保存时出错，请查看后台日志。";
        }});
    }});

    // 页面加载后拉一次 case 列表
    fetchCases();
</script>
</body>
</html>
    """
    return html


@app.route("/api/cases")
def api_cases():
    cases = list_cases()
    return jsonify(
        {
            "cases": [
                {
                    "index": i,
                    "prefix": c["prefix"],
                    "has_encoder": c["has_encoder"],
                }
                for i, c in enumerate(cases)
            ]
        }
    )


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
        encoder_arr = np.copy(sync_arr)  # 默认拿 sync 作为初始 encoder

    dm_img_data_url = image_to_data_url(c["dm_image_path"])
    binary_dm_data_url = image_to_data_url(c["binary_image_path"])

    return jsonify(
        {
            "prefix": c["prefix"],
            "rows": int(rows),
            "cols": int(cols),
            "sync_array": sync_arr.tolist(),
            "encoder_array": encoder_arr.tolist(),
            "has_encoder": c["encoder_array_path"] is not None,
            "dm_image_data_url": dm_img_data_url,
            "binary_dm_data_url": binary_dm_data_url,
        }
    )


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

    # 确保目录存在
    os.makedirs(CHECKED_DIR, exist_ok=True)

    # 存到 checkedcases 目录（npy）
    sync_npy_path = os.path.join(CHECKED_DIR, f"{prefix}_sync_dm_array.npy")
    enc_npy_path = os.path.join(CHECKED_DIR, f"{prefix}_encoder_dm_array.npy")
    np.save(sync_npy_path, sync_arr)
    np.save(enc_npy_path, enc_arr)

    # 生成 sync DM 图（1=白, 0=黑）带白边
    sync_img = dm_array_to_image(sync_arr, cell_w=SYNC_CELL_WIDTH, border=BORDER_WIDTH)
    sync_img_path = os.path.join(CHECKED_DIR, f"{prefix}_sync_dm_code.jpg")
    cv2.imwrite(sync_img_path, sync_img)

    # 生成 encoder DM 图（同样规则）
    enc_img = dm_array_to_image(enc_arr, cell_w=SYNC_CELL_WIDTH, border=BORDER_WIDTH)
    enc_img_path = os.path.join(CHECKED_DIR, f"{prefix}_encoder_dm_code.jpg")
    cv2.imwrite(enc_img_path, enc_img)

    return jsonify(
        {
            "status": "ok",
            "msg": f"sync: {os.path.basename(sync_npy_path)}, encoder: {os.path.basename(enc_npy_path)}",
        }
    )


if __name__ == "__main__":
    print("Running on http://127.0.0.1:5000  (Ctrl+C 退出)")
    app.run(host="127.0.0.1", port=5000, debug=True)
