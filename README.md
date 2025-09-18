# Benchmark

A high-performance C++ multi-threaded benchmark application designed to evaluate CPU performance, memory access patterns, and the efficiency of work-stealing algorithms. It incorporates advanced optimizations such as SIMD (Single Instruction, Multiple Data) for ARM64 and x86_64 architectures, and NUMA (Non-Uniform Memory Access) awareness.

## Features

*   **Multi-threaded Work Processing:** Utilizes a sophisticated work-stealing queue mechanism for efficient task distribution and load balancing across multiple CPU cores.
*   **SIMD Optimizations:** Leverages platform-specific SIMD instruction sets (ARM64 NEON, x86_64 AVX2) for accelerated computation, with a scalar fallback for other architectures.
*   **NUMA Awareness:** Optimizes thread affinity and work distribution to minimize memory latency on NUMA architectures.
*   **Adaptive Work Generation:** Dynamically adjusts the batch size of generated tasks based on system load to maintain optimal queue fullness.
*   **Configurable Scenarios:** Supports predefined benchmark scenarios (throughput, compute, steal) to test different aspects of performance.
*   **Comprehensive Reporting:** Provides detailed performance metrics including operations per second, task completion rates, work-stealing efficiency, and load balance scores.
*   **Command-Line Interface:** Easy-to-use command-line arguments for configuring benchmark parameters such as the number of threads, duration, and scenario.

## Building

To build the benchmark application, you will need a C++17 compatible compiler (e.g., g++).

```bash
g++ -std=c++17 -O3 -Wall -march=native -pthread src/main.cpp -o benchmark
```

*   `-std=c++17`: Specifies the C++17 standard.
*   `-O3`: Enables aggressive compiler optimizations for performance.
*   `-Wall`: Enables all standard warnings.
*   `-march=native`: Optimizes the code for the native architecture of the machine it's being compiled on (enables SIMD instructions like NEON or AVX2 if available).
*   `-pthread`: Links against the POSIX threads library.

## Running

You can run the benchmark with various command-line options:

```bash
./benchmark [options]
```

### Options:

*   `-t, --threads N`: Specifies the number of worker threads to use. If not provided, the benchmark will automatically detect and use the number of available CPU cores.
*   `-d, --duration T`: Sets the duration of the benchmark run in seconds (e.g., `10.0`). Default is 10.0 seconds.
*   `-s, --scenario S`: Selects a predefined benchmark scenario. Available scenarios:
    *   `default`: General-purpose benchmark.
    *   `throughput`: Light work, high frequency tasks.
    *   `compute`: Heavy work, balanced tasks.
    *   `steal`: Light work, aggressive work stealing.
*   `-h, --help`: Displays the help message.

### Examples:

Run with default settings (auto threads, 10 seconds, default scenario):
```bash
./benchmark
```

Run with 8 threads for 30 seconds using the 'compute' scenario:
```bash
./benchmark -t 8 -d 30.0 -s compute
```

Run with 16 threads for 60 seconds using the 'throughput' scenario:
```bash
./benchmark --threads 16 --duration 60.0 --scenario throughput
```

## Performance Expectations

The optimized version of this benchmark is expected to demonstrate:

*   **15-25% higher throughput** due to improved vectorization and efficient work distribution.
*   **Better load balancing** across threads, especially on NUMA systems, from NUMA-aware work distribution.
*   **Reduced contention** through optimized memory ordering and lock-free data structures.
*   **More stable performance** with adaptive algorithms that adjust to system load.

## Contributing

We welcome contributions to improve this benchmark. Please refer to `CONTRIBUTE.md` for detailed guidelines on how to contribute, report issues, and suggest enhancements.

## License

This project is licensed under the [MIT License](LICENSE).
