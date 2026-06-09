# DeDoDe Matcher 升級實作報告

## 背景

目前 `exp_DeDoDe` pipeline 使用 MNN（Mutual Nearest Neighbor）作為 DeDoDe descriptor 的 matcher。MNN 是純幾何方法，沒有任何學習參數，完全依賴 descriptor 的 cosine similarity 做 argmax + mutual check。

本報告規劃兩個實驗：

1. **實驗 A：DualSoftMax ablation**（零成本切換）
2. **實驗 B：LightGlue for DeDoDe-B**（learned matcher 升級）

---

## 實驗 A：`exp_DeDoDe_dualsoftmax`

### 改動範圍

**僅改 pipeline JSON 的 `keypoint_matching_args`**，程式碼零修改。

`match_dedode.py` 已實作 `match_dual_softmax()`，`keypoint_matching_dedode()` 已支援 `match_method="dual_softmax"` 分支。`matching.py` 和 `rotate_matching_find_best.py` 都已透過 `matching_args.get("match_method", "mnn")` 傳遞參數。

### 具體差異（相對於 `exp_DeDoDe`）

| 項目 | `exp_DeDoDe`（原版） | `exp_DeDoDe_dualsoftmax` |
|------|---------------------|--------------------------|
| `match_method` | `"mnn"` | `"dual_softmax"` |
| `mnn_threshold` | `0.85` | 不使用 |
| `dual_softmax_inv_temp` | 不使用 | `20`（預設） |
| `dual_softmax_threshold` | 不使用 | `0.01`（預設） |
| 其餘所有參數 | — | 完全相同 |

### DualSoftMax vs MNN 的行為差異

MNN 的 threshold（0.85）是硬門檻：cosine similarity < 0.85 直接丟棄。DualSoftMax 則是對 similarity 矩陣做雙向 softmax 後取 mutual probability，`inv_temp=20` 控制分佈的銳度，`threshold=0.01` 過濾低置信度匹配。

DualSoftMax 預期產出**更多匹配**（門檻 0.01 遠低於 MNN 的 0.85），但精度依賴後續 RANSAC 過濾。如果匹配數過多，可調高 `dual_softmax_threshold`（建議試 0.05、0.1）。

### 需建立的檔案

```
new-pipelines/exp_DeDoDe_dualsoftmax/
├── pipeline.json
└── transp_pipeline.json
```

---

## 實驗 B：`exp_DeDoDe_lightglue`

### 發現

Kornia 最新版（main branch）的 `LightGlue` 類別已原生支援 DeDoDe-B descriptor：

```python
# kornia/feature/lightglue.py — features dict
"dedodeb": {
    "weights": "dedodeb_lightglue",
    "input_dim": 256,
},
```

權重檔託管在 `http://cmp.felk.cvut.cz/~mishkdmy/models/dedodeb_lightglue.pth`，由 Kornia 維護者 Dmytro Mishkin 訓練。DeDoDe-B 輸出 256 維 descriptor，LightGlue 內部也是 256 維，不需要 projection layer（`input_proj` 是 `nn.Identity()`）。

### 整合架構

現有程式碼的 matcher 分支邏輯：

```
matching.py → task_matching()
├── matcher_type == "LightGlue"
│   ├── detect_keypoints() — ALIKED/DISK/SIFT/SuperPoint
│   └── keypoint_matcing_LG() — KF.LightGlueMatcher(extractor_type, ...)
│
└── matcher_type == "DeDoDe"
    ├── detect_keypoints_dedode() — DeDoDe Detector-L + Descriptor-B
    └── keypoint_matching_dedode() — MNN / DualSoftMax
```

**LightGlue 整合的關鍵接口**是 `keypoint_matcing_LG()`（`match.py`）：

```python
matcher = KF.LightGlueMatcher(extractor_type, matcher_params).eval().to(device)
# ...
distances, indices = matcher(
    descriptors1, descriptors2,
    KF.laf_from_center_scale_ori(keypoints1[None]),
    KF.laf_from_center_scale_ori(keypoints2[None]),
)
```

它需要的輸入是 h5 裡的 keypoints `[N, 2]` 和 descriptors `[N, 256]`，**DeDoDe 分支已經存這些格式**。

### 整合方案

**新增 `matcher_type == "DeDoDe_LightGlue"` 分支**，混合 DeDoDe 提取和 LightGlue 匹配：

1. **特徵提取**：沿用 `detect_keypoints_dedode()`（DeDoDe Detector-L + Descriptor-B，784×784）
2. **匹配**：改用 `KF.LightGlueMatcher("dedodeb", matcher_params)`

這需要在 `matching.py` 和 `rotate_matching_find_best.py` 新增分支。

### 需修改的程式碼

#### 1. `match.py` — 新增 `keypoint_matching_dedode_lightglue()`

```python
def keypoint_matching_dedode_lightglue(
    image_pairs, keypoints_h5_path, descriptions_h5_path,
    matcher_params=None, min_matches=15, verbose=False,
    device=torch.device("cpu"),
):
    if matcher_params is None:
        matcher_params = {"width_confidence": -1, "depth_confidence": -1, "mp": True}
    matcher = KF.LightGlueMatcher("dedodeb", matcher_params).eval().to(device)
    matches = {}
    with h5py.File(keypoints_h5_path, "r") as f_kp, \
         h5py.File(descriptions_h5_path, "r") as f_desc:
        for key1, key2 in tqdm(image_pairs):
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            d1 = torch.from_numpy(f_desc[key1][...]).to(device)
            d2 = torch.from_numpy(f_desc[key2][...]).to(device)
            with torch.inference_mode():
                distances, indices = matcher(
                    d1, d2,
                    KF.laf_from_center_scale_ori(kp1[None]),
                    KF.laf_from_center_scale_ori(kp2[None]),
                )
            n_matches = len(indices)
            if n_matches >= min_matches:
                matches.setdefault(key1, {})
                matches[key1][key2] = indices.detach().cpu().numpy().reshape(-1, 2)
    return matches
```

#### 2. `matching.py` — 新增 `"DeDoDe_LightGlue"` 分支

在 `task_matching()` 中，`elif matcher_type == "DeDoDe_LightGlue":` 分支：
- 特徵提取：複製 `"DeDoDe"` 分支的 `detect_keypoints_dedode()` 邏輯
- 匹配：呼叫 `keypoint_matching_dedode_lightglue()`
- PostProcess：**完全相同**（h5 格式一致）

#### 3. `rotate_matching_find_best.py` — 同樣新增分支

在 DeDoDe 分支的匹配部分，改呼叫 `keypoint_matching_dedode_lightglue()`。

### Pipeline JSON 差異

| 項目 | `exp_DeDoDe` | `exp_DeDoDe_lightglue` |
|------|-------------|------------------------|
| `"matcher"` | `"DeDoDe"` | `"DeDoDe_LightGlue"` |
| `"keypoint_matching_args"` | `match_method`, `mnn_threshold` | `matcher_params`（LightGlue 配置） |
| `dedode_args` | 完全相同 | 完全相同 |
| 其餘 task 結構 | — | 完全相同 |

### LightGlue 參數建議

```json
"keypoint_matching_args": {
    "matcher_params": {
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": true,
        "filter_threshold": 0.1
    },
    "min_matches": 100,
    "verbose": false
}
```

`width_confidence: -1` 和 `depth_confidence: -1` 停用 adaptive pruning 和 early stopping，確保所有 keypoint 都被考慮（與 MNN 行為一致）。`mp: true` 啟用混合精度加速。

### Kornia 版本需求

需要 Kornia >= 0.8.0（`dedodeb` 支援）。目前 Kaggle 環境的 Kornia 版本需確認。如果版本不夠新，有兩個方案：

1. **pip 升級**：`pip install kornia>=0.8.0 --break-system-packages`
2. **手動載入**：直接從 cvg/LightGlue 原始碼用，跳過 Kornia wrapper：
   ```python
   from lightglue import LightGlue
   matcher = LightGlue(features=None, weights="path/to/dedodeb_lightglue.pth", input_dim=256)
   ```
   但需要注意 cvg/LightGlue 的 `features` dict 目前**不包含** `dedodeb`（只有 Kornia fork 有），所以需要用 `features=None` + 手動設定 `input_dim=256`。

### GPU 記憶體影響

| 模型 | 估計 VRAM |
|------|----------|
| LightGlue（9 layers, 4 heads, dim=256） | ~150–300 MB |
| MNN（純矩陣乘法） | ~50 MB（N×M sim 矩陣） |

LightGlue 比 MNN 多用約 200 MB，但 Phase 1 結束後 DeDoDe detector/descriptor 已釋放，Phase 3 的 subprocess 有足夠空間。

### 速度影響

LightGlue 的 transformer 推論比 MNN 的矩陣乘法慢。但在 `num_kp=10000` 的情況下，MNN 的 sim 矩陣是 10000×10000 = 1 億個元素，LightGlue 反而可能透過 early stopping 提早結束。建議先跑 `depth_confidence: -1`（無 early stopping）取得 baseline，再試 `depth_confidence: 0.95` 看速度提升。

---

## 執行順序建議

1. 先跑 **實驗 A（DualSoftMax）**，這是零成本的 ablation，可以快速確認 matcher 改善的天花板
2. 確認 Kaggle 環境的 Kornia 版本，決定 LightGlue 的載入方式
3. 實作程式碼修改（`match.py`、`matching.py`、`rotate_matching_find_best.py`）
4. 跑 **實驗 B（LightGlue）**

---

## Notebook 端需要的改動

### Cell 1 路徑設定

新增：
```python
# Matcher experiment selection
MATCHER_EXPERIMENT = "lightglue"  # "mnn" | "dual_softmax" | "lightglue"
DEDODE_PIPELINE_JSON = f'exp_DeDoDe_{MATCHER_EXPERIMENT}/pipeline.json'  # 或按需指定
DEDODE_TRANSP_PIPELINE_JSON = f'exp_DeDoDe_{MATCHER_EXPERIMENT}/transp_pipeline.json'
```

### Cell 25（DeDoDe Pipeline Patching）

佔位符替換邏輯不變，但需要從對應的實驗目錄讀取 pipeline JSON。

### Cell 26（config.py 覆寫）

指向新的 pipeline JSON 路徑。

### 環境安裝（Cell 2）

如果用 LightGlue 實驗，需確保 Kornia 版本足夠或額外下載 `dedodeb_lightglue.pth` 權重。
