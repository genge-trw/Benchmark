# Project Documentation: Benchmark

This document provides an in-depth look into the architecture, design choices, and operational details of the Benchmark application. It is intended for developers who wish to understand, modify, or extend the project.

## 1. Architecture Overview

The Benchmark application is built around a multi-threaded, work-stealing architecture designed for high-performance CPU and memory stress testing. Key components include:

*   **Worker Threads:** Independent threads that execute work items.
*   **Work Stealing Queues (`WorkStealingQueue`):** Each worker thread has its own local queue. When a thread's local queue is empty, it attempts to "steal" work from other threads' queues.
*   **Work Item Pool (`WorkItemPool`):** A memory pool for `WorkItem` objects to reduce dynamic memory allocation overhead in the hot path.
*   **SIMD Abstraction (`SIMD` namespace):** Provides a platform-agnostic interface for vectorized operations, leveraging NEON on ARM64 and AVX2 on x86_64.
*   **Benchmark Configuration (`BenchmarkConfig`):** A centralized structure for runtime-configurable parameters.
*   **Statistics Collection:** Hot and cold statistics structures (`WorkerStatsHot`, `WorkerStatsCold`) for efficient, cache-aligned performance metric collection.

## 2. Core Components Explained

### 2.1. `WorkItem` Structure

`WorkItem` represents a single unit of work to be performed by a worker thread. It is designed to be cache-friendly and is 32-byte aligned.

```cpp
struct alignas(32) WorkItem {
    uint64_t target_iterations; // Number of iterations for the work function
    uint32_t work_id;           // Unique identifier for the work item
    enum class Type : uint8_t { COMPUTE, SHUTDOWN } type; // Type of work (compute or shutdown signal)
    uint8_t priority;           // For future prioritization (currently unused)
    uint16_t source_thread;    // Tracks the thread that generated this work item
    char padding[8];            // Ensures 32-byte alignment
};
```

### 2.2. `WorkStealingQueue`

This is a custom-built, lock-free (mostly) work-stealing deque implementation. Each worker thread owns one such queue. Threads primarily push and pop from their own `bottom_` (LIFO for cache locality) and steal from other threads' `top_` (FIFO for fairness).

*   **`push(WorkItem* item)`:** Adds a work item to the local queue.
*   **`try_pop(WorkItem*& item)`:** Attempts to pop a work item from the local queue without blocking.
*   **`wait_and_pop(WorkItem*& item)`:** Blocks until a work item is available in the local queue or a shutdown signal is received.
*   **`steal(WorkItem*& item)`:** Attempts to steal a work item from the `top_` of another thread's queue. Includes a simple hazard pointer mechanism to prevent ABA issues.
*   **`shutdown()`:** Signals the queue to shut down, unblocking any waiting threads.

### 2.3. `WorkItemPool`

An optimized memory pool for `WorkItem` objects. This reduces the overhead of frequent `new` and `delete` calls, which can be significant in high-performance scenarios. It uses larger chunks (128 items) for better memory locality and epoch-based reclamation.

*   **`acquire()`:** Retrieves a `WorkItem` from the pool.
*   **`release(WorkItem* item)`:** Returns a `WorkItem` to the pool, resetting its state.
*   **`getWorkItem(...)`:** A convenience method to acquire and initialize a `WorkItem`.
*   **`advance_epoch()`:** Advances the epoch counter, used for memory reclamation.

### 2.4. `SIMD` Namespace

This namespace provides a clean abstraction layer for performing vectorized computations. It conditionally compiles code based on the target architecture (`__aarch64__` for ARM64 NEON, `__x86_64__` for AVX2). If neither is defined, it falls back to a scalar implementation.

*   `performWorkVectorized(uint64_t target_iterations)`: The core vectorized work function that performs a series of arithmetic and bitwise operations on 64-bit integers, optimized for SIMD parallelism.

### 2.5. `BenchmarkConfig`

This struct holds various runtime-configurable parameters that influence the benchmark's behavior. It includes sensible defaults and attempts to auto-detect the cache line size.

```cpp
struct BenchmarkConfig {
    size_t cache_line_size;         // Detected or default cache line size
    uint64_t default_work_iterations; // Default iterations for a single work item
    std::chrono::milliseconds batch_generation_delay; // Delay between work generation batches
    size_t work_item_pool_size;     // Total size of the work item memory pool
    uint64_t update_frequency;      // How often worker threads update their global stats
    size_t max_steal_attempts;      // Maximum attempts a thread makes to steal work
    bool adaptive_stealing;         // Enable/disable adaptive work stealing logic
    bool numa_aware;                // Enable/disable NUMA-aware optimizations
};
```

### 2.6. Statistics Collection

Performance statistics are collected using two structures to minimize false sharing:

*   **`WorkerStatsHot`:** Contains frequently updated, atomic counters (e.g., `operations_completed`, `tasks_processed`, `steals_attempted`, `steals_successful`). It is cache-aligned to 64 bytes.
*   **`WorkerStatsCold`:** Contains less frequently updated or non-atomic data (e.g., `start_time`, `end_time`, `cpu_id`, `stolen_tasks`).

Worker threads update their local `WorkerStatsHot` counters in batches to reduce atomic contention.

## 3. Benchmark Scenarios

The `runBenchmarkScenario` function provides predefined configurations to test specific aspects of the system:

*   **`default`:** A general-purpose benchmark with balanced settings.
*   **`throughput`:** Configured for light work and high frequency task generation, ideal for measuring overall system throughput.
    *   `default_work_iterations = 5000`
    *   `update_frequency = 50`
    *   `max_steal_attempts = 2`
*   **`compute`:** Designed for heavy computational tasks, balancing work distribution.
    *   `default_work_iterations = 50000`
    *   `update_frequency = 200`
    *   `max_steal_attempts = 5`
*   **`steal`:** Focuses on aggressive work stealing, with light work items to encourage frequent stealing attempts.
    *   `default_work_iterations = 1000`
    *   `update_frequency = 20`
    *   `max_steal_attempts = 8`
    *   `adaptive_stealing = true`

## 4. Performance Metrics

The `printResults` function outputs a detailed summary of the benchmark run, including:

*   **Total Operations:** Sum of all computational operations performed.
*   **Total Tasks:** Total number of work items processed.
*   **Operations/second:** Overall throughput of the benchmark.
*   **Average Work Time:** Average time taken to complete a single work item.
*   **Operations per Task:** Average number of operations performed per task.
*   **Work Stealing Statistics:**
    *   Steal Attempts: Total number of times threads tried to steal work.
    *   Successful Steals: Number of times stealing attempts succeeded.
    *   Steal Hit Rate: Percentage of successful steal attempts.
    *   Tasks Stolen: Total number of tasks that were stolen by other threads.
    *   Steal Efficiency: Percentage of total tasks that were stolen.
*   **Load Balance:** A score indicating how evenly work was distributed among threads (calculated as `min_ops / max_ops * 100`).
*   **Platform Info:** Detects and reports the architecture and SIMD optimizations used.

## 5. Known Issues and Potential Improvements

Based on code analysis and previous observations, the following areas have been identified for potential improvement or are known limitations:

*   **Thread ID System Call Issue (`sched_setaffinity`):** The current implementation uses `syscall(SYS_gettid)` which returns a Linux thread ID, but `sched_setaffinity()` expects a process ID. For thread affinity, `pthread_setaffinity_np()` or using `0` for the current thread is more appropriate. (Addressed in `GEMINI.md` and `CONTRIBUTE.md`)
*   **Inconsistent Variable Declaration:** Use of `__i` (double underscore prefix) is reserved for implementation and should be replaced with standard variable names like `i`. (Addressed in `CONTRIBUTE.md`)
*   **False Sharing in `WorkerStats`:** While `WorkerStatsHot` is cache-aligned, atomic operations on different cores might still cause cache line bouncing. Thread-local counters with periodic aggregation could further optimize this.
*   **Missing Error Handling:** Some system calls and operations lack robust error checking.
*   **Resource Management:** Ensure proper thread cleanup and resource deallocation, especially in destructors, to handle exceptions gracefully.
*   **Magic Numbers:** Replace hardcoded numerical values with named constants for better readability and maintainability.
*   **Inconsistent Naming:** Standardize naming conventions (e.g., `camelCase` vs. `snake_case`) across the codebase.
*   **NUMA Awareness:** While basic NUMA awareness is implemented, more advanced NUMA optimizations (e.g., explicit memory allocation on specific NUMA nodes) could be explored.
*   **Lock-Free Queue Implementation:** The current `WorkStealingQueue` is mostly lock-free but uses a fallback mutex for `wait_and_pop`. A fully lock-free, wait-free queue could be considered for extreme contention scenarios.
*   **Memory Layout Optimization:** Further analysis of `WorkItem` and `WorkerStats` memory layouts could yield minor performance gains.

For more detailed analysis and specific code recommendations, refer to `CONTRIBUTE.md`.
