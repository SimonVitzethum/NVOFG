# ADR 0008 — Native DLSS Frame Generation loader ("Path B"): private research, not a product

- **Status:** Decided — **Path B kept as PRIVATE interoperability research; NOT shipped.** The
  shippable product path is the own native model (ADR-tracked as "Plan A", see
  `docs/plan-a-native-framegen.md`).
- **Date:** 2026-07-17
- **Supersedes** the earlier "pursue Path B" framing of this ADR.

## Context

DLSS Frame Generation has no native Linux runtime — it ships only as a Windows PE inside the driver's
Wine/Proton directory, and the native NGX host reports it unsupported. "Path B" explored loading that
module in a native Linux process (no Wine/Proton) so nvofg/RenderFX could interoperate with it.

## What was learned (kept private)

- **Feasibility is proven.** The native driver does support FG, and a native host↔module bridge can
  drive the capability query to the exact result the reference stack gives ("fully supported"). The
  functional path (init → create → evaluate → pacing) is a multi-week reverse-engineering tail.
- The concrete findings are interoperability information obtained by decompilation and are retained
  **locally only**, git-ignored, never published (see Compliance).

## Decision

1. **Retain Path B as private interoperability research** under `experiments/dlssg-native/`, which is
   **git-ignored** (not tracked, not published). It exists to (a) prove feasibility and (b) serve as
   an **offline quality reference** for the own model.
2. **Do not publish or ship Path B**, in any form, and do not distribute NVIDIA binaries.
3. **Build the product on "Plan A"** — an own native frame generator (own/open weights, no NVIDIA
   binary loaded), see `docs/plan-a-native-framegen.md`.

## Compliance rationale

- Reverse engineering **for interoperability** of an independently created program is permitted in
  Germany by **§ 69e UrhG**; the EULA's reverse-engineering ban is **void to that extent** under
  **§ 69g Abs. 2 UrhG**, and **§ 3 Abs. 1 Nr. 2 GeschGehG** permits disassembling a lawfully acquired
  product. The driver is licensed and its files are loaded in place, never copied/redistributed.
- **"Legal in Germany under §69e" is NOT "NVIDIA-compliant."** NVIDIA's EULA forbids the reverse
  engineering, the derivative shim environment, running the module outside its supported runtime, and
  the signature check the loader had to satisfy. No realistic NVIDIA license (including the NGX/DLSS
  SDK license, which blesses only the official SDK integration + redistribution of their *signed*
  runtime) covers a native loader driving the module outside the supported path. So Path B **cannot
  be published NVIDIA-compliant**, and **§ 69e Abs. 2 Nr. 2** independently forbids passing the
  obtained internals to third parties beyond interoperability necessity → it stays private.
- (Even replacing the loader's faked signature assertion with *faithful* Authenticode verification of
  the genuine NVIDIA signature — a cleaner posture, worth doing for the private artifact — removes the
  circumvention concern but not the EULA use/derivative limits, so it does not enable publication.)

## Consequences

- No product coupling to Path B; nvofg/RenderFX ship on Plan A only.
- The only NVIDIA-compliant route to native DLSS-G would be NVIDIA shipping it natively or an explicit
  written agreement — neither is assumed.

## Alternatives

- **Plan A — own native model** (chosen for the product): own/open weights, native from day one, no
  NVIDIA binary, no clearance needed. See `docs/plan-a-native-framegen.md`.
- **Official NGX/DLSS SDK integration** — compliant, but provides no native-Linux Frame Generation
  (the gap this project addresses).
