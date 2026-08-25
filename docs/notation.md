# notation.md — Bảng ký hiệu (bám Table 1, `docs/AUC_SpecDec_Formulation.pdf`)

Nguồn: `AUC_SpecDec_Formulation.tex` (compile ra `docs/AUC_SpecDec_Formulation.pdf`,
7 trang). File này trích nguyên bảng ký hiệu (Table 1, dòng 99–125 của .tex) và bổ
sung equation number để code trace ngược được (mục 4 CLAUDE.md: mọi công thức phải
có comment `# eq. (n)`).

**Lưu ý phạm vi (đọc trước khi dùng bảng này để code):** bản `.tex` hiện tại vẫn mô
tả Problem (P) như bilevel formulation trung tâm (dòng 63–69, 224–259) — đây là
phần **đã bị bỏ** theo scope đã chốt trong CLAUDE.md mục 1 (T7.1/T7.2 trong
TASKS.md sẽ hạ (P) xuống thành 1 remark khi viết lại paper, nhưng bản `.tex` gốc
chưa được sửa). Code trong `sim/` chỉ implement **inner problem (Pₓ)** với ξ cố
định — bỏ qua toàn bộ phần liên quan tới ξ như biến quyết định (`Ξ`, chọn
draft placement/sizing ở outer level).

---

## Bảng ký hiệu

| Ký hiệu | Ý nghĩa | Equation / Definition |
|---|---|---|
| `𝒩`, `N` | tập / số lượng edge stream | §1.1 |
| `𝒟`, `sᵢ`, `m(sᵢ)` | draft library, draft model chọn cho stream i, memory footprint | §1.1 — **ξ, dùng cấu hình cho sẵn, không phải biến tối ưu** |
| `γᵢ(t)`, `γmax` | speculation length của stream i, cận trên | §1.1 |
| `αᵢ` | acceptance rate theo token của stream i (không biết trước) | Assumption 1 |
| `φ(γ,α)` | số token kỳ vọng nhận được mỗi round | eq. (2) — `φ(γ,α) = (1-α^(γ+1))/(1-α)` |
| `Aᵢ(k)` | số token thực nhận ở stream i, round k (accepted + bonus) | Assumption 1 |
| `τᵢ^d` | latency draft mỗi token | §1.1 |
| `κ` | payload uplink mỗi token draft (bits) | §1.1 |
| `wᵢ(t)`, `W`, `rᵢ(t)` | phân bổ băng thông, ngân sách, uplink rate | eq. (1) — Shannon: `rᵢ(t) = wᵢ(t)·log2(1+SNRᵢ(t))`, `Σwᵢ(t) ≤ W` |
| `δᵢ` | fixed per-round latency (downlink + overhead) | §1.1 |
| `𝓑(t)`, `Γ(𝓑)` | verification batch, tổng token count | §1.1 — `Γ(𝓑) = Σᵢ∈𝓑(γᵢ+1)` |
| `θ0`, `θf`, `θm(·)` | per-pass overhead, per-token compute time, memory floor | eq. (3) |
| `bW`, `bKV`, `βmem`, `Lᵢ` | model bytes, KV bytes/token, memory bandwidth, context length | eq. (3) |
| `Γ†` | roofline knee (nơi memory-bound = compute-bound) | eq. (3) — `Γ† = θm(𝓑)/θf` |
| `Wᵢ^q(k)` | queueing delay của stream i, round k | §1.1 |
| `xᵢ`, `Y` | interactivity của stream i, system goodput | eq. (4) (Definition 1) — cả hai công thức chung 1 label `eq:xi` |
| `𝒞(ξ)`, `Y*(x;ξ)`, `x̄(ξ)` | achievable region, frontier, max-min interactivity | eq. (5) (Definition 2, `eq:region`) |
| `AUC(ξ)`, `ÂUC(ξ)` | diện tích dưới frontier, bản chuẩn hóa | eq. (6) (Definition 3, `eq:auc`) — `AUC(ξ) = ∫₀^x̄(ξ) Y*(x;ξ)dx` |
| `ξ∈Ξ`; `π∈Π` | configuration; runtime policy | **ξ cố định trong toàn bộ `sim/`, không sweep** |
| `Zᵢ(t)`, `Q(t)`, `Qᵢ(t)` | virtual queue: interactivity, server-compute, device-time | eq. (12) — dùng trong `auc_controller.py` |
| `V`, `λ(t)`, `μᵢ(t)` | Lyapunov weight, giá verification-token, giá device-time | §1.6 — `λ(t)=Q(t)`, `μᵢ(t)=Qᵢ(t)` |

---

## Equation cross-reference cho `sim/`

Số equation dưới đây lấy trực tiếp từ `\newlabel` trong file `.aux` sau khi compile
(không phải đếm tay) — đảm bảo khớp đúng bản PDF cuối cùng trong `docs/`.

| Eq. | Label (.tex) | Nội dung | Module implement |
|---|---|---|---|
| (1) | `eq:rate-wireless` | Shannon uplink rate | `sim/core/channel.py` |
| (2) | `eq:phi` | `φ(γ,α)` expected tokens/round | `sim/core/acceptance.py` hoặc `sim/metrics/` (dùng chung) |
| (3) | `eq:Tv` | Roofline verify latency `Tᵛ(𝓑)`, knee `Γ†` | `sim/core/server.py` |
| (4) | `eq:xi` | Interactivity `xᵢ` và goodput `Y` (renewal-reward, chung 1 label) | `sim/metrics/frontier.py` |
| (5) | `eq:region` | Achievable region `𝒞(ξ)`, frontier `Y*(x;ξ)`, `x̄(ξ)` | `sim/metrics/frontier.py` |
| (6) | `eq:auc` | `AUC(ξ)`, `ÂUC(ξ)` trapezoidal integral | `sim/metrics/auc.py` |
| (7) | `eq:P` | Outer Problem (P) — **đã bỏ khỏi scope**, KHÔNG implement | — |
| (8) | `eq:Px` | Inner problem (Pₓ) — đây là bài toán duy nhất `sim/` giải | toàn bộ `sim/controllers/` |
| (9) | `eq:bayes` | Bayes-optimal robustness identity (Proposition 1) | không cần code, dùng giải thích kết quả trong paper |
| — | (Proposition 2, dual sweep) | quét η để trace toàn bộ frontier — không có equation riêng, chỉ mô tả bằng lời | `sim/metrics/frontier.py` (sweep V hoặc x) |
| (10) | `eq:Tsym` | `T(B,γ)`, `x(B,γ)` symmetric saturated instance | `sim/run_experiments/exp2_two_regime.py` (Exp 2a) |
| (11) | `eq:gammastar` | `γ*(B)` closed-form (Dinkelbach stopping rule, Theorem 1b) | `sim/run_experiments/exp2_two_regime.py` — so sánh với γ* đo thực nghiệm |
| (12) | `eq:vq` | Virtual queues `Zᵢ(t)`, `Q(t)`, `Qᵢ(t)` | `sim/metrics/virtual_queues.py`, `sim/controllers/auc_controller.py` |
| (13) | `eq:index` | Device-side speculation index `γᵢ(t)` closed form | `sim/controllers/auc_controller.py` — công thức chính |
| (14) | `eq:dppbound` | Theorem 3: `Y ≥ Y*(x;ξ) - C/V`, backlog `O(V)` | validate bằng unit test (TASKS.md T2.4) |
| (15) | `eq:regret` | Theorem 4: `Reg(T) = Õ(√(NT)) + O(T/V)` | validate ở Exp 4 (`exp4_censored_learning.py`) |

Batching (server-side, §1.6-ii — knapsack `max Σωᵢ(t)` theo `Γ(𝒮) ≤ Γbud(t)`, greedy
theo `ωᵢ/(γᵢ+1)`) và radio waterfilling (§1.6-iii — Cauchy–Schwarz square-root
waterfilling) không có số equation riêng trong bản `.tex` hiện tại nhưng đều nằm
trong `auc_controller.py`; comment code nên trỏ về "§1.6(ii)" / "§1.6(iii)" thay vì
số eq. cụ thể.

---

## Assumption cross-reference

| Assumption | Nội dung | Ảnh hưởng tới code |
|---|---|---|
| 1 (Acceptance) | Bernoulli(αᵢ) i.i.d. theo token, αᵢ∈[αmin,αmax] không biết trước | `sim/core/acceptance.py`; kiểm tra độ vững ở `exp_misspecification.py` |
| 2 (Roofline) | `Tᵛ(𝓑)` eq. (3) | `sim/core/server.py` |
| 3 (Wireless) | SNR i.i.d. theo slot, `rmin ≤ rᵢ(t) ≤ rmax` | `sim/core/channel.py` |
| 4 (Closed-loop traffic) | round k+1 phát ngay khi nhận round k — renewal cycle | `sim/core/device.py` |
| 5 (Drafting cost) | draft γ token tốn `γ·τᵢ^d` giây, không overlap với verify | `sim/core/device.py` |
