# QM review package — exact repository subset

Source repository: kolkowitzgroupshared-byte/dioptric  
Source commit: 6fbe4cc788374d00b2c934045685ca775493e436

This package preserves the real repository hierarchy for the NV-array / QM code paths most relevant to review.

## Control panel

`controlpanels/control_panel-purcell.py` is intentionally **trimmed**. It contains only these production entry points:

- `do_widefield_image_sample` — retained because the production spin-echo wrapper calls it before an echo run.
- `do_charge_state_conditional_init`
- `do_adaptive_charge_initialization`
- `do_spin_echo`

Their function bodies are copied verbatim from the source control panel. The unrelated control-panel experiments, sample definitions, analysis helpers, and launch code are omitted.

The production `do_spin_echo()` body references the global `nv_sig` for the pre-run image. That global is defined in the omitted sample-definition portion of the full control panel; this package preserves that reference rather than silently rewriting the production logic.

## Main workflows

### Conditional/adaptive charge preparation
Host camera processing determines charge state and can update the target selection used by the running QUA program.

### Spin echo
The spin-echo experiment does **not** use host feedback during the Hahn-echo sequence. The relevant path is:

`controlpanels/control_panel-purcell.py`
→ `majorroutines/widefield/spin_echo.py`
→ `servers/timing/sequencelibrary/QM_opx/camera/spin_echo.py`
→ `base_scc_sequence.py`
→ `seq_utils.py`

with polarization → π/2 → τ → π → τ → π/2 → SCC → camera readout.

## Verbatim files

Except for the intentionally trimmed control panel and the sanitized config excerpt, the included source files are copied from the source commit without rewriting.

## Configuration

`config/purcell_QM_RELEVANT_EXCERPT.py` retains the relevant production configuration structure while omitting the DeviceIDs/private-network block containing internal addresses, VISA endpoints, cluster identifiers, and instrument serial numbers.
