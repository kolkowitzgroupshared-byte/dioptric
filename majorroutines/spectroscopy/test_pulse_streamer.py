import time

import matplotlib.pyplot as plt
import numpy as np
import TimeTagger as ttag
from pulsestreamer import PulseStreamer

# tt = Tagger.createTimeTagger()

# # pulser.stream(seq, n_runs=5)
# total_count = ttag.Counter(
#     tagger=tt, channels=[1], binwidth=10e-3 / 1e-12, n_values=1000
# )
# time.sleep(10e-3 * 1000)
# # print(total_count.getData())
# data = total_count.getData()[0]
# fig, ax = plt.subplots()
# ax.plot(data)
# plt.show()


### Create the Pulse Sequence
pulser_ts = 1e-9
wait = 1e-6
pulse_dur = 5e-6
decay_period = 1e-6
total_time = pulse_dur + wait + decay_period

pulser = PulseStreamer("192.168.0.111")

# Channel names
optical_ch = 0  # 515 laser
gate_ch = 4

# Digital levels
HIGH = 1
LOW = 0

optical_patt = [
    (50e-3 / pulser_ts, LOW),
    (50e-3 / pulser_ts, HIGH),
    (10e-3 / pulser_ts, LOW),
]
gate_patt = [
    (40e-3 / pulser_ts, LOW),
    (60e-3 / pulser_ts, HIGH),
    (10e-3 / pulser_ts, LOW),
]


# Set channels using class Sequence
seq = pulser.createSequence()
seq.setDigital(optical_ch, optical_patt)
seq.setDigital(gate_ch, gate_patt)


tagger = ttag.createTimeTagger()

start = time.time()
pulser.stream(seq, n_runs=1)
# print(time.time() - start)
# print(time.time() - start)
# time.sleep(1e-5 * 5000)
# print(time.time() - start)

bin_ts = 5e-4
num_bins = int(70e-3 / bin_ts)

hist_1 = ttag.Histogram(
    tagger=tagger,
    click_channel=1,
    start_channel=2,
    binwidth=bin_ts / 1e-12,
    n_bins=num_bins,
)
time.sleep(bin_ts * num_bins)
data = hist_1.getData()

# total_count = ttag.Counter(
#     tagger=tagger, channels=[1], binwidth=bin_ts / 1e-12, n_values=num_bins
# )
# time.sleep(bin_ts * num_bins)
# data = total_count.getData()[0]


fig, ax = plt.subplots()
ax.plot(np.arange(num_bins) * bin_ts, data)
plt.show()
