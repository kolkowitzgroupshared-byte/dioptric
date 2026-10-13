# QM / OPX1000 review package

Source machine: NVOffice
Source repository: C:\Users\saroj\Github\dioptric
Source commit: bc290d8da6e529cdea0679bda9470e54b05fb8f0

Working-tree status before package-generation artifacts:
M analysis/sc_resonance_analysis_optimized.py

The pre-existing working-tree modification is not included unless its path
appears in the exact-file list.

## Workflows

1. Conditional/adaptive charge preparation:
   host camera processing -> charge-state estimate -> runtime target selection
   -> selective QUA polarization. Adaptive mode can additionally update DMD
   blocking from the host.

2. Widefield Hahn echo:
   ordinary spin experiment with no host feedback inside the Hahn sequence:
   polarization -> pi/2 -> tau -> pi -> tau -> pi/2 -> SCC -> camera readout.

## Control panel

controlpanels/control_panel-purcell.py is intentionally curated but preserves:
- current NVSig / nv_list construction
- active NV-coordinate loading from config
- pixel -> green/red AOD coordinate transformation
- per-NV polarization/SCC amplitude calculation
- threshold field construction
- per-NV polarization/ionization/SCC durations
- representative-NV selection
- relevant experiment entry points

The retained function bodies and NVSig-construction block come from the current
NVOffice control panel. Unrelated experiments are omitted.

## Relevant servers

- servers/mixed/QM_opx.py
- servers/inputs/camera_NUVU_hnu512gamma.py
- servers/outputs/dmd_DLP6500.py
- servers/timing/sequencelibrary/QM_opx/... QUA sequence/macro layer

## Core utilities

The package includes constants.py, widefield.py, positioning.py, common.py,
tool_belt.py, data_manager.py, kplotlib.py, and majorroutines/targeting.py.

## Configuration

config/purcell_QM_SANITIZED.py is the current Purcell configuration with only
the DeviceIDs block redacted. Private network addresses, VISA endpoints, COM
ports, serial numbers, and the current QM host/cluster identifiers are omitted.
Optical, microwave, camera, positioning, server mapping, wiring, and OPX
configuration remain present.

## Exact vs curated

Every file listed in SOURCE_SHA256.txt is a byte-for-byte copy of the current
NVOffice checkout.

Generated/curated files:
- controlpanels/control_panel-purcell.py
- config/purcell_QM_SANITIZED.py
- README_QM.md
- checksum manifests

