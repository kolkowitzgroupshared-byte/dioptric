"""Cornell-Berkeley Experiment 0: OPX camera sequence.

The production sequence and the standalone waveform simulation both use the
standard widefield SCC pipeline:

    seq_utils.init() -> macro_run_aods() ->
    base_scc_sequence.macro(
        polarization -> arbitrary microwave gate -> tomography rotation ->
        SCC -> charge readout + camera trigger
    )

Per gate the production experiment measures Z, X, Y and an alternating
ms=0/ms=1 dual-rail SCC reference. The standalone simulation uses the SAME
get_seq() and base_scc_sequence.macro(), with shortened spin-polarization and
charge-readout durations so the full first shot is easy to inspect.

Run from the dioptric repo root:
    python -m servers.timing.sequencelibrary.QM_opx.camera.arbitrary_gate_tomography

Note: after the first camera readout, the real sequence waits for a camera
trigger. Without a simulated incoming trigger, the OPX waveform simulation
naturally stops advancing at this wait. This is the same camera handshake as
the hardware experiment; the first shot still shows the FULL SCC pipeline.

OPX waveform simulation does NOT model NV populations or gate fidelity.
"""

from qm import qua

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(
    base_scc_seq_args,
    compiled_gates,
    num_reps=1,
    *,
    spin_pol_duration_override_cc=None,
    readout_duration_override_cc=None,
):
    """Build the normal production QUA sequence.

    Existing positional calls get_seq(base_args, gates, num_reps) are unchanged.
    Duration overrides are optional and only used by __main__ simulation.
    """
    if num_reps < 2:
        raise ValueError("Dual-rail SCC reference needs at least two reps")
    mw_chains = base_scc_seq_args[-1]
    if isinstance(mw_chains, int):
        mw_chains = [mw_chains]
    if len(mw_chains) != 1:
        raise ValueError("Experiment 0 requires one microwave chain")
    if not compiled_gates:
        raise ValueError("No compiled gates supplied")

    for gate in compiled_gates:
        for p in gate["pulses"]:
            if p["phase_deg"] not in (0, 90, 180, 270) or int(p["duration_cc"]) < 4:
                raise ValueError(f"Invalid compiled pulse: {p}")

    def make_macro(pulses, basis):
        def uwave_macro(uwave_ind_list, step_val):
            qua.align()
            # Cornell SU(2) gate: chronological XYX/YXY resonant pulses.
            for p in pulses:
                seq_utils.macro_pi_pulse(
                    uwave_ind_list,
                    duration_cc=int(p["duration_cc"]),
                    phase=int(p["phase_deg"]),
                )

            # Rotate X and Y measurement axes to Z for standard SCC readout.
            if basis == "X":
                seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=270)
            elif basis == "Y":
                seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            elif basis != "Z":
                raise ValueError(f"Unknown tomography basis: {basis}")
            qua.align()

        return uwave_macro

    uwave_macros = [
        make_macro(gate["pulses"], basis)
        for gate in compiled_gates
        for basis in ("Z", "X", "Y")
    ]

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        # Same polarization -> microwave -> SCC -> camera routine as Rabi,
        # bootstrap tomography, and other standard Purcell camera sequences.
        base_scc_sequence.macro(
            base_scc_seq_args,
            uwave_macros,
            num_reps=num_reps,
            reference=True,
            spin_pol_duration_override=spin_pol_duration_override_cc,
            readout_duration_override=readout_duration_override_cc,
        )

    return seq, []


if __name__ == "__main__":
    import copy
    import traceback
    from pathlib import Path

    import matplotlib.pyplot as plt
    from qm import QuantumMachinesManager
    from qm.simulate import SimulationConfig

    from utils import common
    from utils import tool_belt as tb
    from utils.single_qubit_gate_compiler import compile_gates

    # ------------------------------------------------------------------
    # Standalone FULL-SCC simulation settings (no real output generated).
    # ------------------------------------------------------------------
    SIM_UWAVE_IND = 1       # Working IQ generator on Purcell
    SIM_GATE_INDEX = 0      # Cornell gate index, 0 through 9
    SIM_NUM_REPS = 2        # Build even/odd dual-rail reference branches
    SIM_DURATION_NS = 150_000
    SIM_SPIN_POL_NS = 10_000
    SIM_READOUT_NS = 20_000

    config_module = common.get_config_module()
    config = config_module.config
    opx_config = copy.deepcopy(config_module.opx_config)

    # Same waveform-simulation setup used in camera/rabi.py.
    tb.set_delays_to_zero(opx_config)
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 10e3

    repo_root = Path(__file__).resolve().parents[5]
    csv_file = (
        repo_root / "calibrations" / "purcell" /
        "arbitrary_gate_tomography" / "arbSQR.csv"
    )
    pi_ns = config["Microwaves"]["VirtualSigGens"][SIM_UWAVE_IND]["pi_pulse"]
    manifest = compile_gates(csv_file, pi_ns, SIM_UWAVE_IND)
    gate = manifest["gates"][SIM_GATE_INDEX]

    print(f"Simulating Cornell gate {gate['gate']} on microwave chain {SIM_UWAVE_IND}")
    print(f"Configured pi pulse: {pi_ns} ns")
    print("Gate pulses (phase degrees, duration ns):")
    print([(p["phase_deg"], p["duration_ns"]) for p in gate["pulses"]])
    print("Sequence: polarization -> gate -> Z tomography -> SCC -> readout")
    print("The first camera-trigger wait stops further shots without input triggers.")

    # Standard 8-element base_scc_seq_args structure. Two example NVs;
    # coordinates/durations are simulation inputs and do not move real AODs.
    base_scc_seq_args = [
        [[107.715, 107.724], [107.438, 105.983]],  # 520-nm AOD xy, MHz
        [132, 132],                                # charge prep, ns
        [1.0, 1.0],                                # charge prep amplitude
        [[72.439, 73.236], [72.175, 71.822]],    # 638-nm AOD xy, MHz
        [40, 88],                                  # SCC duration, ns
        [1.0, 1.0],                                # SCC amplitude
        [False, False],                            # spin-flip mask
        [SIM_UWAVE_IND],                            # IQ microwave generator
    ]

    # IMPORTANT: simulate the actual get_seq()/base_scc_sequence.macro(),
    # NOT a separate microwave-only approximation of the experiment.
    seq, _ = get_seq(
        base_scc_seq_args,
        [gate],
        num_reps=SIM_NUM_REPS,
        spin_pol_duration_override_cc=seq_utils.convert_ns_to_cc(
            SIM_SPIN_POL_NS
        ),
        readout_duration_override_cc=seq_utils.convert_ns_to_cc(
            SIM_READOUT_NS
        ),
    )

    try:
        qmm = QuantumMachinesManager(**config["DeviceIDs"]["QM_opx_args"])
        sim = qmm.simulate(
            opx_config,
            seq,
            SimulationConfig(
                duration=seq_utils.convert_ns_to_cc(SIM_DURATION_NS)
            ),
        )
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.suptitle(
            f"Cornell gate {gate['gate']}: init / pol / MW / SCC / readout"
        )
        plt.show(block=True)
    except Exception:
        traceback.print_exc()
        raise

    # Deliberately do not call close_all_quantum_machines(): simulation
    # should not shut down an unrelated experiment on the shared OPX.
