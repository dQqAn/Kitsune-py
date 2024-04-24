from Kitsune import Kitsune
import numpy as np
import time
import pickle
import psutil

# from line_profiler import profile
# from line_profiler.explicit_profiler import profile

# Make a function with the following: process.cpu_percent, memory_info, consider memory_percent maybe?,
# The file where we store the memory usage logs
fp = open('memorytest.log', 'w+')

# Get the process of the current program
process = psutil.Process()

##############################################################################
# Kitsune a lightweight online network intrusion detection system based on an ensemble of autoencoders (kitNET).
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates Kitsune's ability to incrementally learn, and detect anomalies in recorded a pcap of the Mirai Malware.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

# The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 3.6.3   #######################

# Load Mirai pcap (a recording of the Mirai botnet malware being activated)
# The first 70,000 observations are clean...
print("Unzipping Sample Capture...")
import zipfile

with zipfile.ZipFile("mirai.zip", "r") as zip_ref:
    zip_ref.extractall()

# File location
path = "mirai.pcap"  # the pcap, pcapng, or tsv file to process.
packet_limit = np.Inf  # the number of packets to process

# KitNET params:
maxAE = 10  # maximum size for any autoencoder in the ensemble layer
FMgrace = 5000  # the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 20000  # the number of instances used to train the anomaly detector (ensemble itself)

# How often to display the number of processed packets
display_freq = 1000
last_packets = 30000

# Call cpu_percent to measure how much CPU is used to build Kitsune
process.cpu_percent()
# Measure RAM usage before starting Kitsune
ram_before = process.memory_info().vms

# Build Kitsune
K = Kitsune(path, packet_limit, maxAE, FMgrace, ADgrace)

# Measure RAM usage after building Kitsune
ram_after = process.memory_info().vms

# Measure the CPU percentage while building Kitsune
fp.write("CPU percentage used while building Kitsune: " + str(process.cpu_percent()) + "\n")
# Measure RAM after building Kitsune
fp.write("RAM used while building Kitsune: " + str(ram_after - ram_before) + "\n")

print("Running Kitsune:")
# liste veya dizideki kök ortalama kare hatası (Root Mean Square Error - RMSE)
RMSEs = []
i = 0

# Call cpu_percent to measure how much CPU is used to process packets
process.cpu_percent()
# Measure RAM usage before processing packets
ram_before = process.memory_info().vms

start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
# Processing the packets
while True:
    i += 1
    if i % display_freq == 0:
        print(f"Packet {i} and time taken: ", time.time() - start)
    rmse = K.proc_next_packet()
    if i > last_packets:
        break
    if rmse == -1:
        break
    RMSEs.append(rmse)
stop = time.time()
print("Complete. Time elapsed: " + str(stop - start))

# Measure RAM usage after processing packets
ram_after = process.memory_info().vms

# Measure the CPU percentage while processing packets
fp.write("CPU percentage used while processing packets: " + str(process.cpu_percent()) + "\n")
# Measure RAM after processing packets
fp.write("RAM used while processing packets: " + str(ram_after - ram_before) + "\n")

# Save the results in a pickle file
pickle.dump(RMSEs, open("test_results_test.p", "wb"))
print("All packets have been processed. Time elapsed: " + str(stop - start))

### for figure use the figure_genarator.py or u can use this
# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
from scipy.stats import norm

benignSample = np.log(RMSEs[FMgrace + ADgrace + 1:100000])
logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

# plot the RMSE anomaly scores
print("Plotting results")
from matplotlib import pyplot as plt
from matplotlib import cm

plt.figure(figsize=(10, 5))
fig = plt.scatter(range(FMgrace + ADgrace + 1, len(RMSEs)), RMSEs[FMgrace + ADgrace + 1:], s=0.1,
                  c=logProbs[FMgrace + ADgrace + 1:], cmap='RdYlGn')
plt.yscale("log")
plt.title("Anomaly Scores from Kitsune's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Time elapsed [min]")
figbar = plt.colorbar()
figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.show()
