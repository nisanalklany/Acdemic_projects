# Operating Systems Project – Parallel Data Processing

## Overview
This project focuses on efficient processing of student grade datasets using parallel programming techniques.

The system combines **multithreading** and **multiprocessing** to improve performance and reduce execution time when handling large data files.

---

## Objectives
- Accelerate data processing using parallelism
- Efficiently compute statistical metrics on large datasets
- Utilize system resources (CPU cores) effectively

---

## Approach

### 🔹 Multithreading (I/O Operations)
- Used threads to read multiple grade files in parallel
- Each file represents a different subject
- Improved I/O performance by concurrent file reading

### 🔹 Multiprocessing (Computation)
- Used multiple processes to perform heavy statistical computations
- Each process handles a portion of the dataset
- Leveraged all CPU cores for parallel computation

### 🔹 Synchronization
- Implemented **Locks** to prevent race conditions
- Used **Queues** for safe data transfer between processes

---

## Features
- Parallel file reading (multithreading)
- Parallel statistical computation (multiprocessing)
- Calculation of:
  - Mean
  - Median
  - Grade distribution (histogram)
- Automatic generation of summary report

---

## Output
The system generates a summary file:
