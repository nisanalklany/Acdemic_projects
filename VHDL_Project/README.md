# VHDL Project – Matrix Multiplication System (FPGA)

## Overview
Designed and implemented a hardware-based matrix multiplication system using VHDL on FPGA.

The system multiplies two 4×4 signed matrices and displays the result in real-time using a 7-segment display interface.

---

## Objectives
- Implement efficient matrix multiplication in hardware
- Design a modular and scalable architecture
- Optimize memory access and parallel computation
- Control the system using Finite State Machines (FSM)

---

## System Architecture

The design consists of several key components:

### 🔹 Main Controller (FSM)
- Controls the entire system flow:
  - Data acquisition
  - Computation
  - Display
- Implements multiple state machines for control and computation

### 🔹 Memory (RAM)
- Stores input matrices and result matrix
- Supports synchronous read/write operations
- Uses byte-enable for efficient data access

### 🔹 Parallel Multipliers
- Four parallel multipliers compute partial products simultaneously
- Supports signed 8-bit inputs
- Configurable latency

### 🔹 Data Path
- Row-by-column multiplication
- Accumulation of partial products
- Result stored back into memory

---

## Features
- Parallel computation using multiple multipliers
- FSM-based control logic
- Efficient memory management
- Real-time result display using 7-segment
- Support for signed arithmetic

---

## Operation Flow
1. System waits in idle state
2. Receives first matrix (16 elements)
3. Receives second matrix (16 elements)
4. Performs matrix multiplication
5. Stores results in memory
6. Displays results sequentially upon user request

---

## Results
- Successfully implemented full matrix multiplication in hardware
- Achieved correct and stable results in simulation
- Demonstrated efficient use of FPGA resources

---

## Key Learnings
- Designing complex systems using FSM
- Hardware parallelism vs sequential execution
- Memory organization and data flow optimization
- Timing considerations and latency handling in digital systems

---

## Files
- SRAM_Final_Project_Report.pdf – Full report including architecture, FSM design, and simulation results.
## Grade : 97/100
