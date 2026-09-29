"""V17 production rerank: V14 background with shared visibility cap raised 3 -> 5.

Reuses the complete V6 candidate pool and the validated V14 beta=2, taper=0
reranking machinery.  The only physics change is VISIBILITY_SCALE_MAX=5.
"""
from __future__ import annotations

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14

v6.VISIBILITY_SCALE_MAX = 5.0
v14.MODEL_TAG = "v17_beta2_taper0_smax5"

if __name__ == "__main__":
    v14.main()
