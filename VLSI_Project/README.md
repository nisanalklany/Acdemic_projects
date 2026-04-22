# VLSI Project – SRAM Design (Full Custom)

## Overview
Designed and implemented a full-custom SRAM memory system using Cadence Virtuoso, starting from transistor-level design up to layout and system-level integration.

The project focused on building a stable and high-performance SRAM architecture, including all supporting peripheral circuits.

---

## Objectives
- Design a reliable 6T SRAM cell
- Implement a complete memory subsystem
- Perform full-custom design (schematic → layout → verification)
- Analyze performance, timing, and stability

---

## System Architecture

The system includes the following components:

- 6T SRAM memory cell
- Precharge circuits
- Sense amplifiers
- Row/column decoders
- Tri-state buffers
- Latches and control logic

---

## Design Approach

### 🔹 Transistor-Level Design
- Designed a stable 6T SRAM cell
- Optimized transistor sizing for:
  - Read stability
  - Write ability
  - Noise margins

### 🔹 Peripheral Circuits
- Implemented precharge and sense amplifier circuits
- Designed decoders for memory addressing
- Integrated tri-state buffers and latches for data control

### 🔹 Layout & Verification
- Created full-custom layout using Cadence Virtuoso
- Performed:
  - DRC (Design Rule Check)
  - LVS (Layout vs Schematic)
- Ensured layout matches schematic functionality

### 🔹 Simulation
- Simulated circuits using Spectre
- Analyzed:
  - Bitline discharge behavior
  - Timing delays
  - Dynamic operation

---

## Features
- Full-custom VLSI design
- Complete SRAM datapath implementation
- Dual clock phase operation
- High-Z control logic
- Verified layout (DRC & LVS clean)

---

## Results
- Successfully implemented an 8×8 SRAM memory array
- Achieved stable read/write operations
- Demonstrated correct functionality under simulation
- Identified timing constraints and performance bottlenecks

---

## Key Learnings
- Deep understanding of SRAM design and stability
- Transistor sizing trade-offs (performance vs robustness)
- Importance of timing synchronization in datapath design
- Practical experience with Cadence Virtuoso (schematic, layout, simulation)
- Debugging complex circuits using waveform analysis

---

## Challenges
- Ensuring signal integrity across the layout
- Managing timing between different modules
- Integrating multiple blocks into a single working system

---

## Files
- SRAM_Final_Project_Report.pdf – Full report including schematics, layout, and simulation results
