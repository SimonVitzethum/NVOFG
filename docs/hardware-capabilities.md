# OFA hardware capabilities (as probed)

Captured by `src/spike/probe.cpp` on the reference machine. These are the concrete
numbers the pipeline design is built against; re-run the probe on other GPUs before
relying on them.

## Reference machine
- GPU: **NVIDIA GeForce RTX 5070 Laptop** (Blackwell)
- Driver: **610.43.03**, Vulkan 1.4
- Second device present (Intel iGPU) — **no** `VK_NV_optical_flow`; nvofg must select
  the NVIDIA device explicitly.

## `VkPhysicalDeviceOpticalFlowPropertiesNV`

| Field | Value | Consequence for nvofg |
|---|---|---|
| `opticalFlow` feature | true | OFA usable |
| `hintSupported` | **1** | App motion vectors can be fed as OFA **hints** (req. #3 fusion happens *inside* the OFA, not only in post) |
| `costSupported` | **1** | Per-cell **cost map** available → confidence weighting (req. #2) |
| `bidirectionalFlowSupported` | **1** | N→N-1 flow in HW → occlusion/disocclusion (req. #1) |
| `globalFlowSupported` | **1** | Global-motion hint available (camera pan / reset heuristics) |
| output grid sizes | 1×1, 2×2, 4×4 | Finest is **per-pixel** flow; pick per quality tier |
| hint grid sizes | 1×1, 2×2, 4×4, 8×8 | Hints can be coarser than output |
| min / max resolution | 32×32 / **8192×8192** | Covers all real present resolutions |
| `maxNumRegionsOfInterest` | 32 | Optional ROI-restricted flow (future optimisation) |

## OFA image formats (per usage)

`vkGetPhysicalDeviceOpticalFlowImageFormatsNV` reports:

| Usage | Formats | Notes |
|---|---|---|
| **input** / reference | `B8G8R8A8_UNORM` (44), `R8_UNORM` (9), `G8_B8R8_2PLANE_420_UNORM` (1000156003 = NV12) | Format-prep converts the app's color frame into one of these. R8 (luma) is the cheapest; NV12 keeps some chroma. |
| **output** (flow) | `R16G16_SFIXED5_NV` (1000464000) | Flow vectors: 2× signed 16-bit, **5 fractional bits** → range ±1024 px, precision **1/32 px**. Refine/warp must decode S10.5. |
| **hint** | `R16G16_SFIXED5_NV` | App MVs must be converted to the same S10.5 pixel-space encoding to be used as hints. |
| **cost** | `R32_UINT` (98), `R8_UINT` (13) | Cost/confidence map. R8 is enough for a weight; lower cost = higher confidence. |

## Queues
- A **dedicated optical-flow queue family** exists (index 5 on this GPU), distinct from
  graphics/compute. OFA execute is submitted there; results are consumed by compute on
  the graphics/compute queue. → nvofg needs **cross-queue timeline-semaphore** sync and
  correct queue-family ownership transfers for the flow/cost images.

## Design implications (summary)
1. **Hints are first-class.** Because `hintSupported=1`, motion-vector fusion (req. #3)
   is done in two complementary places: (a) feed app MVs as an OFA *hint* to steer the
   hardware search, and (b) fuse again in `refine` using the cost map as the blend
   weight. This is strictly better than post-only fusion.
2. **Cost drives confidence, not a hard threshold** (req. #2): map cost → `confidence ∈
   [0,1]` and use it as a continuous blend/hole weight.
3. **Bidirectional in HW** (req. #1): request forward + backward; forward/backward
   disagreement + depth gives occlusion and disocclusion masks.
4. **S10.5 decode** everywhere flow is read; ±1024 px is ample for 60→120 fps deltas.
5. **Two-queue sync** is a first-order design constraint, not an afterthought.
