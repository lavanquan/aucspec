# model_choices.md — Draft/target model cho calibrate simulator (TASKS.md T0.4)

## Quyết định

**Target**: `Qwen/Qwen2.5-7B-Instruct`
**Draft library `𝒟`** (2 cỡ, EXPERIMENTS.md yêu cầu 2–3 cỡ): `Qwen/Qwen2.5-0.5B-Instruct`,
`Qwen/Qwen2.5-1.5B-Instruct`

Dùng để **profile tham số roofline/draft-latency đưa vào simulator** (T4.1),
không phải để chạy testbed thật trên Setonix ngay — xem "Testbed thật" bên dưới.

## Lý do

### 1. Sửa một lỗi kỹ thuật trong ví dụ ở CLAUDE.md/TASKS.md

TASKS.md T0.4 gợi ý ví dụ *"Llama-3.2-1B / Qwen2.5-0.5B làm draft, Llama-3-8B làm
target"`. Ví dụ này **không dùng được trực tiếp**: speculative decoding chuẩn
(rejection sampling theo token, Assumption 1 trong formulation) đòi hỏi draft và
target **chia sẻ đúng một tokenizer/vocabulary** — nếu không, không thể so token-id
draft với token-id target ở từng vị trí để chấp nhận/từ chối. Llama-3 dùng tokenizer
riêng (vocab 128,256, tiktoken-based) khác hẳn Qwen2.5 (vocab 151,936, Qwen BPE) —
ghép Qwen2.5-0.5B (draft) với Llama-3-8B (target) sẽ **không chạy được** như một hệ
speculative decoding thật, chỉ hợp lệ nếu dùng cơ chế "draft độc lập tokenizer" phức
tạp hơn nhiều (không nằm trong scope formulation hiện tại — Assumption 1 giả định
acceptance Bernoulli theo token, ngầm định cùng vocab).

→ Quyết định: **giữ draft/target trong cùng 1 họ model, cùng tokenizer.**

### 2. Vì sao chọn họ Qwen2.5 thay vì Llama-3

- **Tái sử dụng được pipeline đã chạy thật**: `legacy/src/qwen_edge_specsim/` (prototype
  trước khi restructure, xem [legacy/](../legacy/)) đã có sẵn code vLLM batched
  candidate-verification hoạt động đúng với `Qwen2.5-7B-Instruct` (target) +
  `Qwen2.5-1.5B-Instruct` (draft) — xem `configs/models.yaml` cũ (nay đã archive).
  Đây là hạ tầng thật đã kiểm chứng, không phải chọn mới từ đầu.
- **Cùng tokenizer trong toàn họ Qwen2.5** (0.5B/1.5B/7B đều dùng chung vocab
  151,936) — draft/target ghép hợp lệ ở mọi cặp kích thước.
- **Thêm Qwen2.5-0.5B-Instruct** vào draft library để có 2 cỡ draft (0.5B, 1.5B) —
  đáp ứng yêu cầu "Draft library D: 2–3 cỡ" của EXPERIMENTS.md, cho phép Exp 1/2 so
  sánh trade-off giữa draft nhỏ (rẻ, α thấp hơn) và draft lớn (đắt, α cao hơn).
- **Kích thước target 7B** đủ lớn để roofline có cả 2 regime (memory-bound/
  compute-bound, Theorem 1) rõ ràng ở batch size thực tế (B~8–16, khớp N trong
  `configs/base.yaml`), và đủ nhỏ để profile nhanh trên 1 GPU.

### 3. Testbed thật: chưa chạy trên Setonix, ưu tiên simulation trước

CLAUDE.md mục 3 đã chốt: *"Ưu tiên discrete-event simulator trước..., sau đó nếu có
phần cứng thật thì thêm testbed layer."* Quyết định T0.4 này áp dụng đúng nhánh đó:

- Setonix GPU node (`sinfo -p gpu`) là **AMD MI250X** (`gpu:8` mỗi node, ROCm stack),
  trong khi `environment.yml` hiện pin `vllm==0.9.1` cài qua `pip` — bản wheel PyPI
  mặc định build cho **CUDA**, không chạy thẳng trên ROCm. Pipeline vLLM cũ trong
  `legacy/` cũng khai `draft_devices: [cuda:1, cuda:2, cuda:3]` — tức là đã chạy
  trên máy NVIDIA khác trước đây, không phải trên Setonix.
- Vì vậy: **profile draft/target model thật (T4.1) nên chạy trên máy NVIDIA sẵn có**
  (nơi pipeline `legacy/` đã chạy được), chỉ export số đo (τᵈ, θ0, θf, βmem, phân
  phối α) làm input cho `configs/base.yaml`/`configs/models.yaml` mới — simulator
  trong `sim/` tự nó không cần GPU, chạy được trên Setonix CPU node bình thường.
- Nếu sau này build testbed thật trên Setonix, cần build vLLM từ source với ROCm
  backend (`VLLM_TARGET_DEVICE=rocm`) — ghi chú lại đây để không lặp lại nhầm lẫn.

### 4. N edge device / draft placement (Remark 5, EXPERIMENTS.md)

EXPERIMENTS.md gợi ý "Pi5-class / Orin-class giả lập" cho N=8–16 edge device khi có
phần cứng thật. Không nằm trong T0.4 (đây là quyết định về *draft/target model*,
không phải *edge hardware*) — để lại cho T4.1 khi chuẩn bị trace thật.

## Bảng tóm tắt

| Vai trò | Model | Vocab size | Ghi chú |
|---|---|---|---|
| Target | `Qwen/Qwen2.5-7B-Instruct` | 151,936 | roofline profile theo eq. (3) |
| Draft (nhỏ) | `Qwen/Qwen2.5-0.5B-Instruct` | 151,936 | draft rẻ, kỳ vọng α thấp hơn |
| Draft (vừa) | `Qwen/Qwen2.5-1.5B-Instruct` | 151,936 | đã có pipeline chạy thật trong `legacy/` |

## Việc tiếp theo phụ thuộc quyết định này

- T4.1: profile τᵈ (draft latency), θ0/θf/βmem (roofline) trên máy NVIDIA có sẵn,
  điền số thật vào `configs/base.yaml` (thay các giá trị đánh dấu `PROFILE`).
- T4.2: log accept/reject thật theo token cho cặp draft/target ở trên, làm input
  `sim/core/acceptance.py` (mode `trace_driven`) và oracle α̂ᵢ (T2.3).
- `configs/models.yaml` mới (khác bản legacy đã archive) sẽ trỏ đúng 3 model này khi
  Phase 4 bắt đầu — chưa tạo ở bước T0.4 này vì chưa có số đo thật để điền.
