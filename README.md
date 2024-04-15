105.263 mm
113.98 mm

## Pruning random resistive memory for optimizing analog AI


The rapid advancement of artificial intelligence (AI) has been marked by the large language models exhibiting human-like
intelligence. However, these models also present unprecedented challenges to energy consumption and environmental
sustainability, which are exacerbated by the increasing number of users and the development of even larger models. One
promising solution is to revisit analog computing, a technique that predates digital computing and exploits emerging analog
electronic devices, such as resistive memory, which features in-memory computing, high scalability, and nonvolatility that
addresses the von Neumann bottleneck, slowdown of Moore’s law, and volatile DRAM of conventional digital hardware. However,
analog computing still faces the same challenges as before: programming nonidealities and expensive programming due to the
underlying device’s physics. Therefore, leveraging the efficiency advantage while mitigating the programming disadvantage of
analog computing with resistive memory is a major open problem in AI hardware and electronics communities. Here, we report
a universal solution, software-hardware co-design using structural plasticity-inspired edge pruning to optimize the topology of
a randomly weighted analog resistive memory neural network. Software-wise, the topology of a randomly weighted neural
network is optimized by pruning connections rather than precisely tuning resistive memory weights. Hardware-wise, we reveal
the physical origin of the programming stochasticity using transmission electron microscopy, which is leveraged for large-scale
and low-cost implementation of an overparameterized random neural network containing high-performance sub-networks. We
implemented the co-design on a 40nm 256K resistive memory macro, observing 17.3% and 19.9% accuracy improvements in
image and audio classification on FashionMNIST and Spoken Digit datasets, as well as 9.8% (2%) improvement in PR (ROC)
in image segmentation on DRIVE datasets, respectively. This is accompanied by 82.1%, 51.2%, and 99.8% improvement
in energy efficiency thanks to analog in-memory computing. Finally, we demonstrated the generality of our co-design with
alternative analog memory technologies and scalability using ResNet-50 for ImageNet100 learning. By embracing the intrinsic
stochasticity and in-memory computing, this work may solve the biggest obstacle of analog computing systems and thus
unleash their immense potential for next-generation AI hardware.


### Hardware and Software Information
#### Hardware
```
GPU: NVIDIA GeForce RTX 4090
CPU: AMD EPYC 7542 32-Core Processor
Memory: 128G
```


#### Software
```
Python version:  3.8.18
[GCC 9.4.0]
torch version:  2.1.0
CUDA is available
CUDA version:  12.1
```


### Code information 
In this code, we have two main components: the "xxx_software_TO" part and the "xxx_hardware_TO" part. The "xxx_software_TO" part of the code is designed to run on a traditional CPU+GPU architecture, providing software baseline results for our topology optimization (TO) method. This section aims to establish a solid foundation for comparison with the hardware implementation. Next, we have the "xxx_hardware_TO" part of the code, which simulates the implementation process of the Processing-in-Memory (PIM) system in a Python environment. This allows you to easily replicate our experimental results without accessing the actual hardware PIM platform. By utilizing the conductance readout data that we have extracted from a 256K RRAM chip, you can effectively recreate our experiments and observe the performance of the PIM system in action. You can download the hardware weights from the following link:
```
https://connecthkuhk-my.sharepoint.com/:u:/g/personal/songqi27_connect_hku_hk/EUvMGLv225JOrxiP8Bpnwa8BdCyVHWharOS2kUZOWfm76A?e=NRRv9G
```
After downloading the hardware weights, you need to put them in the same directory as the code, i.e. xxxxx/CNN_FashionMNIST_TO/CNN_RRAM_hardware_weight.pt.


### RRAM PIM chip information 
Under the 40nm technology node, the fabricated resistive memory chip features a 512×512 crossbar array, with resistive
memory cells constructed between the metal 4 and metal 5 layers using the backend-of-line process. These cells comprise
bottom and top electrodes (BE and TE) and a transition-metal oxide dielectric layer. The BE via, which possesses a diameter of
60nm, undergoes patterning through photo-lithography and etching. It is then filled with TaN via physical vapor deposition,
followed by an overlay of a 10nm TaN buffer layer. Subsequently, a 5nm Ta layer is deposited and oxidized, resulting in
the creation of an 8nm TaOx dielectric layer. The TE is formed by sequentially depositing 3nm Ta and 40nm TiN through
physical vapor deposition. Upon completion of fabrication, the remaining interconnection metals are deposited using the
standard logic process. The resistive memory cells in the same row share BE connections, while those in the same column
share TE connections. After undergoing a 30-minute post-annealing process at 400°C in a vacuum, the 40nm resistive memory
chip demonstrates exceptional performance, exhibiting high yield and robust endurance characteristics.


### Hybrid Analog–Digital Hardware System information 
The hybrid analog-digital hardware system (see Supplementary Information Fig. 3) consists of a 40nm resistive memory chip
and a Xilinx ZYNQ system-on-chip (SoC), which includes a field-programmable gate array (FPGA) and advanced RISC
machines (ARM) processor integrated on a printed circuit board (PCB). The resistive memory chip operates in three primary
modes according to the edge pruning topology optimization: electroform mode for generating random conductance weights,
reset mode for pruning selected weights, and multiplication mode for vector-matrix products. The electroform mode triggers a
dielectric breakdown in resistive memory arrays and forms random conductance matrices, with All source lines (SLs) biased to
a fixed programming voltage sourced by an eight-channel digital-to-analog converter (DAC, DAC80508, Texas Instruments)
with 16-bit resolution, while bit lines (BLs) are grounded and word lines (WLs) are biased by the DAC to impose compliance
current to cells and prevent hard breakdown. The SL voltage amplitude and width modulate the post-breakdown conductance
distribution and sparsity. The reset mode restores a resistive memory cell back to its off-state, with the selected BL biased
by the DAC, while the selected SL is grounded and the rest of the SLs float. For multiplication mode, a 4-channel analog
multiplexer (CD4051B, Texas Instruments) with an 8-bit shift register (SN74HC595, Texas Instruments) applies a DC voltage
to the BLs of the resistive memory chip. During each training step, the resistive memory chip is read, and the multiplication
values carried by the current from the SLs are converted to voltages using trans-impedance amplifiers (OPA4322-Q1, Texas
Instruments) and analog-to-digital converters (ADS8324, Texas Instruments, 14-bit resolution). These results are then sent to
the Xilinx SoC for further processing. The FPGA contains logic that drives resistive memory and exchanges data with the
ARM processor using direct double-data rate memory access via the direct memory access control unit. Additionally, the FPGA
implements some neural network functions in hardware, such as activation and pooling.
