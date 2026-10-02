# Exact NV / QM repository subset

This package preserves the production repository hierarchy for the files most relevant to Quantum Machines review.

Source checkout: C:\Users\saroj\Github\dioptric
Git commit: 6fbe4cc788374d00b2c934045685ca775493e436

## Important distinction

- Conditional/adaptive charge preparation contains host-to-running-QUA feedback.
- Hahn/spin_echo does NOT use host feedback during the spin experiment.
- spin_echo.py is included to show the real polarization -> pi/2 - tau - pi - tau - pi/2 -> SCC -> camera workflow.

## Verbatim production files

All files under majorroutines/, servers/, and controlpanels/ in this package are byte-for-byte copies from the current checkout.

## Config

config/purcell_QM_RELEVANT_EXCERPT.py is the only excerpt. It begins after the DeviceIDs/private-network block, which was deliberately excluded before sharing externally. The production lines that are included were not rewritten.

## Current working-tree status

Only unrelated analysis files were modified when this package was generated:
 M analysis/sc_deer_hahn.py
 M analysis/sc_deer_hahn_p1_pulse_position_scan.py
?? share_with_qm/qm_exact_repo_subset_2026-09-30/

Those modified analysis files are NOT included in this package.
