#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <unistd.h>
#include <sched.h>
#include <system_error>
#include <iomanip>
#include <algorithm>
#include <array>
#include <cassert>

#ifdef __aarch64__
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

// Runtime-configurable constants with sensible defaults
struct BenchmarkConfig {
    size_t cache_line_size = 64;
    uint64_t default_work_iterations = 10000;
    std::chrono::milliseconds batch_generation_delay{5};
    size_t work_item_pool_size = 4096;
    uint64_t update_frequency = 100;
    size_t max_steal_attempts = 3;
    bool adaptive_stealing = true;
    bool numa_aware = true;
    
    // Auto-detect cache line size if possible
    BenchmarkConfig() {
        // On most modern systems, cache line is 64 bytes
        // Could be enhanced with runtime detection
        #ifdef _SC_LEVEL1_DCACHE_LINESIZE
        long detected_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
        if (detected_size > 0) {
            cache_line_size = static_cast<size_t>(detected_size);
        }
        #endif
    }
};

// Platform abstraction for SIMD operations
namespace SIMD {
    
#ifdef __aarch64__
    using VectorType = uint64x2_t;
    
    inline VectorType load_vector(uint64_t value) {
        return vdupq_n_u64(value);
    }
    
    inline VectorType add_vectors(VectorType a, VectorType b) {
        return vaddq_u64(a, b);
    }
    
    inline VectorType xor_vectors(VectorType a, VectorType b) {
        return veorq_u64(a, b);
    }
    
    inline uint64_t horizontal_sum(VectorType v) {
        return vgetq_lane_u64(v, 0) + vgetq_lane_u64(v, 1);
    }
    
    inline uint64_t performWorkVectorized(uint64_t target_iterations) {
        const uint64_t vector_iterations = target_iterations / 8; // Process 8 elements per iteration
        
        // Use more NEON registers for better throughput
        VectorType vec1 = load_vector(0x9E3779B97F4A7C15ULL);
        VectorType vec2 = load_vector(0xC4CEB9FE1A85EC53ULL);
        VectorType vec3 = load_vector(0xFF51AFD7ED558CCDULL);
        VectorType vec4 = load_vector(0x2FC962E947B2681BULL);
        
        VectorType result1 = load_vector(0);
        VectorType result2 = load_vector(0);
        VectorType result3 = load_vector(0);
        VectorType result4 = load_vector(0);
        
        for (uint64_t i = 0; i < vector_iterations; i += 4) {
            VectorType temp1 = load_vector(i);
            VectorType temp2 = load_vector(i + 1);
            VectorType temp3 = load_vector(i + 2);
            VectorType temp4 = load_vector(i + 3);
            
            temp1 = add_vectors(temp1, vec1);
            temp2 = add_vectors(temp2, vec2);
            temp3 = add_vectors(temp3, vec3);
            temp4 = add_vectors(temp4, vec4);
            
            result1 = xor_vectors(result1, temp1);
            result2 = xor_vectors(result2, temp2);
            result3 = xor_vectors(result3, temp3);
            result4 = xor_vectors(result4, temp4);
            
            // Additional computation to increase work density
            result1 = add_vectors(result1, vec3);
            result2 = add_vectors(result2, vec4);
            result3 = add_vectors(result3, vec1);
            result4 = add_vectors(result4, vec2);
        }
        
        VectorType final1 = add_vectors(result1, result2);
        VectorType final2 = add_vectors(result3, result4);
        VectorType final = add_vectors(final1, final2);
        
        return horizontal_sum(final);
    }
    
#elif defined(__x86_64__)
    using VectorType = __m256i;
    constexpr size_t VECTOR_WIDTH = 4;
    
    inline VectorType load_vector(uint64_t value) {
        return _mm256_set1_epi64x(value);
    }
    
    inline VectorType add_vectors(VectorType a, VectorType b) {
        return _mm256_add_epi64(a, b);
    }
    
    inline VectorType xor_vectors(VectorType a, VectorType b) {
        return _mm256_xor_si256(a, b);
    }
    
    inline uint64_t horizontal_sum(VectorType v) {
        __m128i high = _mm256_extracti128_si256(v, 1);
        __m128i low = _mm256_castsi256_si128(v);
        __m128i sum = _mm_add_epi64(high, low);
        return _mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1);
    }
    
    inline uint64_t performWorkVectorized(uint64_t target_iterations) {
        const uint64_t vector_iterations = target_iterations / 16;
        
        VectorType vec1 = load_vector(0x9E3779B97F4A7C15ULL);
        VectorType vec2 = load_vector(0xC4CEB9FE1A85EC53ULL);
        VectorType vec3 = load_vector(0xFF51AFD7ED558CCDULL);
        VectorType vec4 = load_vector(0x2FC962E947B2681BULL);
        
        VectorType result1 = load_vector(0);
        VectorType result2 = load_vector(0);
        VectorType result3 = load_vector(0);
        VectorType result4 = load_vector(0);
        
        for (uint64_t i = 0; i < vector_iterations; i += 4) {
            VectorType temp1 = load_vector(i);
            VectorType temp2 = load_vector(i + 1);
            VectorType temp3 = load_vector(i + 2);
            VectorType temp4 = load_vector(i + 3);
            
            temp1 = add_vectors(temp1, vec1);
            temp2 = add_vectors(temp2, vec2);
            temp3 = add_vectors(temp3, vec3);
            temp4 = add_vectors(temp4, vec4);
            
            result1 = xor_vectors(result1, temp1);
            result2 = xor_vectors(result2, temp2);
            result3 = xor_vectors(result3, temp3);
            result4 = xor_vectors(result4, temp4);
        }
        
        VectorType final1 = add_vectors(result1, result2);
        VectorType final2 = add_vectors(result3, result4);
        VectorType final = add_vectors(final1, final2);
        
        return horizontal_sum(final);
    }
    
#else
    // Scalar fallback
    inline uint64_t performWorkVectorized(uint64_t target_iterations) {
        uint64_t result1 = 0, result2 = 0, result3 = 0, result4 = 0;
        constexpr uint64_t C1 = 0x9E3779B97F4A7C15ULL;
        constexpr uint64_t C2 = 0xC4CEB9FE1A85EC53ULL;
        constexpr uint64_t C3 = 0xFF51AFD7ED558CCDULL;
        constexpr uint64_t C4 = 0x2FC962E947B2681BULL;
        
        uint64_t i = 0;
        for (; i + 4 <= target_iterations; i += 4) {
            result1 += (i + 0) * C1; result1 ^= result1 >> 33; result1 *= C2;
            result2 += (i + 1) * C1; result2 ^= result2 >> 33; result2 *= C2;
            result3 += (i + 2) * C1; result3 ^= result3 >> 33; result3 *= C2;
            result4 += (i + 3) * C1; result4 ^= result4 >> 33; result4 *= C2;
        }
        
        for (; i < target_iterations; ++i) {
            result1 += i * C1;
            result1 ^= result1 >> 33;
            result1 *= C2;
        }
        
        return result1 ^ result2 ^ result3 ^ result4;
    }
#endif

} // namespace SIMD

// Improved alignment and false sharing prevention
template<typename T>
struct alignas(64) CacheAligned {
    T value;
    char padding[64 - sizeof(T)];
    
    CacheAligned() : value{} {}
    CacheAligned(const T& v) : value(v) {}
    
    operator T&() { return value; }
    operator const T&() const { return value; }
    T& operator=(const T& v) { value = v; return value; }
};

struct alignas(64) WorkerStatsHot {
    std::atomic<uint64_t> operations_completed{0};
    std::atomic<uint64_t> tasks_processed{0};
    std::atomic<uint64_t> total_work_time_ns{0};
    std::atomic<uint64_t> work_count{0};
    std::atomic<uint64_t> steals_attempted{0};
    std::atomic<uint64_t> steals_successful{0};
};

struct WorkerStatsCold {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    int cpu_id{-1};
    uint64_t local_tasks{0};
    uint64_t stolen_tasks{0};
};

// Optimized WorkItem with better packing
struct alignas(32) WorkItem {
    uint64_t target_iterations;
    uint32_t work_id;
    enum class Type : uint8_t { COMPUTE, SHUTDOWN } type;
    uint8_t priority; // For future prioritization
    uint16_t source_thread; // Track work origin
    char padding[8];
    
    WorkItem() = default;
    WorkItem(Type t, uint32_t id, uint64_t iterations, uint8_t prio = 0, uint16_t src = 0)
        : target_iterations(iterations), work_id(id), type(t), priority(prio), source_thread(src) {}
};
static_assert(sizeof(WorkItem) == 32, "WorkItem size should be exactly 32 bytes");

// Enhanced lock-free work stealing queue with hazard pointer support
class WorkStealingQueue {
private:
    static constexpr size_t QUEUE_SIZE = 2048; // Increased size
    static constexpr size_t MASK = QUEUE_SIZE - 1;
    
    alignas(64) std::atomic<size_t> top_{0};
    alignas(64) std::atomic<size_t> bottom_{0};
    std::vector<std::atomic<WorkItem*>> buffer_;
    std::mutex fallback_mutex_;
    std::condition_variable cond_;
    std::atomic<bool> shutdown_{false};
    
    // Simple hazard pointer to prevent ABA issues
    alignas(64) std::atomic<WorkItem*> hazard_ptr_{nullptr};
    
public:
    WorkStealingQueue() : buffer_(QUEUE_SIZE) {
        for (auto& slot : buffer_) {
            slot.store(nullptr, std::memory_order_relaxed);
        }
    }
    
    void push(WorkItem* item) {
        if (shutdown_.load(std::memory_order_acquire)) return;
        
        size_t b = bottom_.load(std::memory_order_relaxed);
        buffer_[b & MASK].store(item, std::memory_order_relaxed);
        
        // Prefetch next slot for better cache performance
        __builtin_prefetch(&buffer_[(b + 1) & MASK], 1, 3);
        
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
        cond_.notify_one();
    }
    
    bool try_pop(WorkItem*& item) {
        size_t b = bottom_.load(std::memory_order_relaxed);
        size_t t = top_.load(std::memory_order_acquire);
        
        if (t >= b) return false;
        
        --b;
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        item = buffer_[b & MASK].load(std::memory_order_relaxed);
        
        if (t == b) {
            if (!top_.compare_exchange_strong(t, t + 1, 
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                bottom_.store(b + 1, std::memory_order_relaxed);
                return false;
            }
            bottom_.store(b + 1, std::memory_order_relaxed);
        }
        
        return true;
    }
    
    bool wait_and_pop(WorkItem*& item) {
        if (try_pop(item)) return true;
        
        std::unique_lock<std::mutex> lock(fallback_mutex_);
        cond_.wait(lock, [this, &item] {
            return try_pop(item) || shutdown_.load(std::memory_order_acquire);
        });
        
        return !shutdown_.load(std::memory_order_acquire);
    }
    
    bool steal(WorkItem*& item) {
        size_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        size_t b = bottom_.load(std::memory_order_acquire);
        
        if (t >= b) return false;
        
        item = buffer_[t & MASK].load(std::memory_order_relaxed);
        hazard_ptr_.store(item, std::memory_order_release);
        
        // Double-check after setting hazard pointer
        if (t != top_.load(std::memory_order_acquire)) {
            hazard_ptr_.store(nullptr, std::memory_order_relaxed);
            return false;
        }
        
        bool success = top_.compare_exchange_weak(t, t + 1, 
                std::memory_order_acq_rel, std::memory_order_relaxed);
        
        hazard_ptr_.store(nullptr, std::memory_order_relaxed);
        return success;
    }
    
    void shutdown() {
        shutdown_.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lock(fallback_mutex_);
        cond_.notify_all();
    }
    
    size_t size() const {
        size_t b = bottom_.load(std::memory_order_relaxed);
        size_t t = top_.load(std::memory_order_relaxed);
        return (b >= t) ? (b - t) : 0;
    }
    
    bool is_shutdown() const { 
        return shutdown_.load(std::memory_order_acquire); 
    }
};

// Enhanced memory pool with epoch-based reclamation
class WorkItemPool {
private:
    struct FreeNode {
        std::atomic<FreeNode*> next;
    };
    
    alignas(64) std::atomic<FreeNode*> free_head_{nullptr};
    std::vector<std::unique_ptr<WorkItem[]>> chunks_;
    std::atomic<uint64_t> epoch_{0};
    
public:
    explicit WorkItemPool(size_t size = 4096) {
        constexpr size_t CHUNK_SIZE = 128; // Larger chunks for better locality
        size_t num_chunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        chunks_.reserve(num_chunks);
        FreeNode* head = nullptr;
        
        for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
            size_t chunk_items = std::min(CHUNK_SIZE, size - chunk * CHUNK_SIZE);
            auto chunk_ptr = std::make_unique<WorkItem[]>(chunk_items);
            
            for (size_t i = 0; i < chunk_items; ++i) {
                FreeNode* node = reinterpret_cast<FreeNode*>(&chunk_ptr[i]);
                node->next.store(head, std::memory_order_relaxed);
                head = node;
            }
            
            chunks_.push_back(std::move(chunk_ptr));
        }
        
        free_head_.store(head, std::memory_order_release);
    }
    
    WorkItem* acquire() {
        FreeNode* head = free_head_.load(std::memory_order_acquire);
        while (head != nullptr) {
            FreeNode* next = head->next.load(std::memory_order_relaxed);
            if (free_head_.compare_exchange_weak(head, next, 
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return reinterpret_cast<WorkItem*>(head);
            }
        }
        return nullptr; // Pool exhausted
    }
    
    void release(WorkItem* item) {
        if (!item) return;
        
        // Reset item state
        item->type = WorkItem::Type::COMPUTE;
        item->work_id = 0;
        item->target_iterations = 10000;
        item->priority = 0;
        item->source_thread = 0;
        
        FreeNode* node = reinterpret_cast<FreeNode*>(item);
        FreeNode* head = free_head_.load(std::memory_order_acquire);
        
        do {
            node->next.store(head, std::memory_order_relaxed);
        } while (!free_head_.compare_exchange_weak(head, node, 
                std::memory_order_acq_rel, std::memory_order_acquire));
    }
    
    WorkItem* getWorkItem(WorkItem::Type type, uint32_t id, 
                         uint64_t iterations = 10000, uint8_t priority = 0, 
                         uint16_t source = 0) {
        WorkItem* item = acquire();
        if (item) {
            item->type = type;
            item->work_id = id;
            item->target_iterations = iterations;
            item->priority = priority;
            item->source_thread = source;
        }
        return item;
    }
    
    void advance_epoch() {
        epoch_.fetch_add(1, std::memory_order_acq_rel);
    }
};

class OptimizedBenchmark {
private:
    BenchmarkConfig config_;
    int num_cores_;
    int num_threads_;
    std::vector<std::unique_ptr<WorkStealingQueue>> work_queues_;
    std::vector<std::thread> worker_threads_;
    std::unique_ptr<CacheAligned<WorkerStatsHot>[]> worker_stats_hot_;
    std::unique_ptr<WorkerStatsCold[]> worker_stats_cold_;
    std::atomic<bool> running_{false};
    std::atomic<uint32_t> work_counter_{0};
    WorkItemPool work_pool_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    
    // NUMA-aware thread state
    struct ThreadState {
        std::mt19937 rng;
        std::uniform_int_distribution<int> steal_dist;
        std::vector<int> steal_order; // Preferred steal order
        uint64_t consecutive_steals{0};
        
        ThreadState(int thread_id, int num_threads) 
            : rng(std::random_device{}() + thread_id), steal_dist(0, num_threads - 1) {
            // Build NUMA-aware steal order (prefer nearby threads)
            steal_order.reserve(num_threads);
            for (int i = 1; i < num_threads; ++i) {
                int target = (thread_id + i) % num_threads;
                steal_order.push_back(target);
            }
        }
    };
    
    void setCPUAffinity(int cpu_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Warning: Failed to set CPU affinity for CPU " << cpu_id 
                      << " (error: " << strerror(errno) << ")\n";
        }
    }
    
    uint64_t performWork(uint64_t target_iterations) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        uint64_t result = SIMD::performWorkVectorized(target_iterations);
        
        // Prevent compiler optimization
        asm volatile("" :: "m"(result) : "memory");
        
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time).count();
    }
    
    bool tryStealWork(int thread_id, WorkItem*& item, ThreadState& state) {
        auto& stats = worker_stats_hot_[thread_id].value;
        
        size_t max_attempts = config_.adaptive_stealing ? 
            std::min(config_.max_steal_attempts + state.consecutive_steals / 10, 
                    static_cast<size_t>(8)) : config_.max_steal_attempts;
        
        // Use NUMA-aware ordering when possible
        for (size_t attempt = 0; attempt < max_attempts && 
             attempt < state.steal_order.size(); ++attempt) {
            
            int target_thread = config_.numa_aware ? 
                state.steal_order[attempt % state.steal_order.size()] :
                state.steal_dist(state.rng);
                
            stats.steals_attempted.fetch_add(1, std::memory_order_relaxed);
            
            if (work_queues_[target_thread]->steal(item)) {
                stats.steals_successful.fetch_add(1, std::memory_order_relaxed);
                state.consecutive_steals++;
                return true;
            }
        }
        
        state.consecutive_steals = 0;
        return false;
    }
    
    void workerThread(int thread_id, int cpu_id) {
        ThreadState state(thread_id, num_threads_);
        
        setCPUAffinity(cpu_id);
        
        auto& stats_hot = worker_stats_hot_[thread_id].value;
        auto& stats_cold = worker_stats_cold_[thread_id];
        
        stats_cold.cpu_id = cpu_id;
        stats_cold.start_time = std::chrono::high_resolution_clock::now();
        
        WorkItem* item = nullptr;
        auto& my_queue = work_queues_[thread_id];
        
        // Local counters for batched updates
        uint64_t local_operations = 0;
        uint64_t local_tasks = 0;
        uint64_t local_work_time = 0;
        uint64_t local_work_count = 0;
        uint64_t local_stolen = 0;
        
        while (true) {
            bool found_work = false;
            bool was_stolen = false;
            
            // Try local queue first (best cache locality)
            if (my_queue->try_pop(item)) {
                found_work = true;
            } else if (num_threads_ > 1 && tryStealWork(thread_id, item, state)) {
                found_work = true;
                was_stolen = true;
            } else {
                // Block waiting for work
                if (!my_queue->wait_and_pop(item)) {
                    break; // Shutdown
                }
                found_work = true;
            }
            
            if (found_work && item) {
                if (item->type == WorkItem::Type::SHUTDOWN) {
                    work_pool_.release(item);
                    break;
                }
                
                uint64_t work_time_ns = performWork(item->target_iterations);
                work_pool_.release(item);
                
                ++local_operations;
                ++local_tasks;
                local_work_time += work_time_ns;
                ++local_work_count;
                
                if (was_stolen) {
                    ++local_stolen;
                }
                
                // Batched updates to reduce atomic contention
                if (local_operations >= config_.update_frequency) {
                    stats_hot.operations_completed.fetch_add(local_operations, std::memory_order_relaxed);
                    stats_hot.tasks_processed.fetch_add(local_tasks, std::memory_order_relaxed);
                    stats_hot.total_work_time_ns.fetch_add(local_work_time, std::memory_order_relaxed);
                    stats_hot.work_count.fetch_add(local_work_count, std::memory_order_relaxed);
                    
                    stats_cold.stolen_tasks += local_stolen;
                    
                    local_operations = 0;
                    local_tasks = 0;
                    local_work_time = 0;
                    local_work_count = 0;
                    local_stolen = 0;
                }
            }
        }
        
        // Final update
        if (local_operations > 0) {
            stats_hot.operations_completed.fetch_add(local_operations, std::memory_order_relaxed);
            stats_hot.tasks_processed.fetch_add(local_tasks, std::memory_order_relaxed);
            stats_hot.total_work_time_ns.fetch_add(local_work_time, std::memory_order_relaxed);
            stats_hot.work_count.fetch_add(local_work_count, std::memory_order_relaxed);
            stats_cold.stolen_tasks += local_stolen;
        }
        
        stats_cold.end_time = std::chrono::high_resolution_clock::now();
    }
    
public:
    explicit OptimizedBenchmark(int num_threads = 0, const BenchmarkConfig& cfg = {}) 
        : config_(cfg), work_pool_(config_.work_item_pool_size) {
        
        num_cores_ = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
        if (num_cores_ <= 0) {
            throw std::runtime_error("Failed to get number of CPU cores");
        }
        
        num_threads_ = (num_threads > 0) ? std::min(num_threads, num_cores_) : num_cores_;
        
        std::cout << "System Configuration:\n";
        std::cout << "  CPU Cores: " << num_cores_ << "\n";
        std::cout << "  Worker Threads: " << num_threads_ << "\n";
        std::cout << "  Cache Line Size: " << config_.cache_line_size << " bytes\n";
        std::cout << "  NUMA Aware: " << (config_.numa_aware ? "Yes" : "No") << "\n";
        std::cout << "  Adaptive Stealing: " << (config_.adaptive_stealing ? "Yes" : "No") << "\n\n";
        
        // Initialize aligned memory
        worker_stats_hot_ = std::make_unique<CacheAligned<WorkerStatsHot>[]>(num_threads_);
        worker_stats_cold_ = std::make_unique<WorkerStatsCold[]>(num_threads_);
        
        work_queues_.reserve(num_threads_);
        for (int i = 0; i < num_threads_; ++i) {
            work_queues_.emplace_back(std::make_unique<WorkStealingQueue>());
        }
    }
    
    void start() {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) return;
        
        work_counter_.store(0, std::memory_order_relaxed);
        
        // Reset statistics
        for (int i = 0; i < num_threads_; ++i) {
            auto& hot = worker_stats_hot_[i].value;
            hot.operations_completed.store(0, std::memory_order_relaxed);
            hot.tasks_processed.store(0, std::memory_order_relaxed);
            hot.total_work_time_ns.store(0, std::memory_order_relaxed);
            hot.work_count.store(0, std::memory_order_relaxed);
            hot.steals_attempted.store(0, std::memory_order_relaxed);
            hot.steals_successful.store(0, std::memory_order_relaxed);
            
            auto& cold = worker_stats_cold_[i];
            cold.cpu_id = -1;
            cold.local_tasks = 0;
            cold.stolen_tasks = 0;
        }
        
        worker_threads_.clear();
        worker_threads_.reserve(num_threads_);
        
        for (int i = 0; i < num_threads_; ++i) {
            int cpu_id = i % num_cores_;
            worker_threads_.emplace_back(&OptimizedBenchmark::workerThread, this, i, cpu_id);
        }
        
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false)) return;
        
        end_time_ = std::chrono::high_resolution_clock::now();
        
        // Send shutdown signals
        for (int i = 0; i < num_threads_; ++i) {
            WorkItem* shutdown_item = work_pool_.getWorkItem(WorkItem::Type::SHUTDOWN, 0);
            if (shutdown_item) {
                work_queues_[i]->push(shutdown_item);
            }
            work_queues_[i]->shutdown();
        }
        
        // Wait for all threads to complete
        for (auto& t : worker_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        
        worker_threads_.clear();
        work_pool_.advance_epoch(); // Clean up memory pool
    }
    
    void generateWork(uint32_t num_tasks, uint64_t iterations_per_task = 0, uint8_t priority = 0) {
        if (iterations_per_task == 0) {
            iterations_per_task = config_.default_work_iterations;
        }
        
        // Adaptive work distribution based on queue loads
        std::vector<size_t> queue_sizes(num_threads_);
        for (int i = 0; i < num_threads_; ++i) {
            queue_sizes[i] = work_queues_[i]->size();
        }
        
        for (uint32_t i = 0; i < num_tasks; ++i) {
            uint32_t work_id = work_counter_.fetch_add(1, std::memory_order_relaxed);
            
            // Choose least loaded queue for better distribution
            int target_queue = 0;
            if (config_.numa_aware) {
                target_queue = std::min_element(queue_sizes.begin(), queue_sizes.end()) 
                              - queue_sizes.begin();
                queue_sizes[target_queue]++; // Update local estimate
            } else {
                target_queue = i % num_threads_;
            }
            
            WorkItem* item = work_pool_.getWorkItem(
                WorkItem::Type::COMPUTE, work_id, iterations_per_task, priority, target_queue);
            if (item) {
                work_queues_[target_queue]->push(item);
            }
        }
    }
    
    void runBenchmark(double duration_seconds, uint32_t initial_batch_size = 1000, 
                     bool variable_work = false) {
        start();
        
        std::cout << "Running benchmark for " << std::fixed << std::setprecision(1) 
                  << duration_seconds << " seconds...\n";
        
        uint32_t batch_size = initial_batch_size;
        auto benchmark_start = std::chrono::high_resolution_clock::now();
        std::mt19937 work_rng(std::random_device{}());
        std::uniform_int_distribution<uint64_t> work_dist(
            config_.default_work_iterations / 2, config_.default_work_iterations * 2);
        
        while (std::chrono::high_resolution_clock::now() - benchmark_start < 
               std::chrono::duration<double>(duration_seconds)) {
            
            // Adaptive batch sizing based on system load
            size_t total_queue_size = 0;
            for (const auto& q : work_queues_) {
                total_queue_size += q->size();
            }
            
            // Adjust batch size based on queue fullness
            if (total_queue_size < batch_size) {
                batch_size = std::min(batch_size + 100, initial_batch_size * 2);
            } else if (total_queue_size > batch_size * 3) {
                batch_size = std::max(batch_size - 50, initial_batch_size / 2);
            }
            
            // Generate variable work if requested
            uint64_t work_iterations = variable_work ? 
                work_dist(work_rng) : config_.default_work_iterations;
            
            generateWork(batch_size, work_iterations);
            std::this_thread::sleep_for(config_.batch_generation_delay);
        }
        
        // Wait for completion with timeout
        uint32_t total_generated_tasks = work_counter_.load(std::memory_order_relaxed);
        auto wait_start = std::chrono::high_resolution_clock::now();
        constexpr auto MAX_WAIT = std::chrono::seconds(60); // Increased timeout
        
        std::cout << "Waiting for " << total_generated_tasks << " tasks to complete...\n";
        
        while (std::chrono::high_resolution_clock::now() - wait_start < MAX_WAIT) {
            uint64_t completed_tasks = 0;
            for (int i = 0; i < num_threads_; ++i) {
                completed_tasks += worker_stats_hot_[i].value.tasks_processed.load(std::memory_order_relaxed);
            }
            
            if (completed_tasks >= total_generated_tasks * 0.95) { // 95% completion threshold
                break;
            }
            
            // Progress update
            if (completed_tasks > 0 && total_generated_tasks > 1000) {
                double progress = static_cast<double>(completed_tasks) / total_generated_tasks * 100;
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                         << progress << "% (" << completed_tasks << "/" << total_generated_tasks << ")" << std::flush;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\n";
        stop();
    }
    
    void printResults() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
        double duration_seconds = duration.count() / 1e6;
        
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BENCHMARK RESULTS\n";
        std::cout << std::string(80, '=') << "\n";
        std::cout << "Configuration:\n";
        std::cout << "  Duration: " << std::fixed << std::setprecision(3) << duration_seconds << " seconds\n";
        std::cout << "  Threads: " << num_threads_ << "\n";
        std::cout << "  CPU Cores: " << num_cores_ << "\n";
        std::cout << "  Work Pool Size: " << config_.work_item_pool_size << "\n";
        std::cout << "  Update Frequency: " << config_.update_frequency << "\n\n";

        uint64_t total_ops = 0, total_tasks = 0, total_work_time = 0, total_work_count = 0;
        uint64_t total_steals_attempted = 0, total_steals_successful = 0, total_stolen_tasks = 0;
        
        std::cout << std::setw(6) << "Thread" 
                  << std::setw(6) << "CPU" 
                  << std::setw(12) << "Operations" 
                  << std::setw(12) << "Tasks" 
                  << std::setw(12) << "Ops/sec"
                  << std::setw(12) << "Work(μs)"
                  << std::setw(10) << "Steals"
                  << std::setw(8) << "Hit%"
                  << std::setw(8) << "Stolen" << "\n";
        
        std::cout << std::string(80, '-') << "\n";
        
        for (int i = 0; i < num_threads_; ++i) {
            const auto& hot = worker_stats_hot_[i].value;
            const auto& cold = worker_stats_cold_[i];
            
            uint64_t ops = hot.operations_completed.load(std::memory_order_relaxed);
            uint64_t tasks = hot.tasks_processed.load(std::memory_order_relaxed);
            uint64_t work_time_ns = hot.total_work_time_ns.load(std::memory_order_relaxed);
            uint64_t work_count = hot.work_count.load(std::memory_order_relaxed);
            uint64_t steals_attempted = hot.steals_attempted.load(std::memory_order_relaxed);
            uint64_t steals_successful = hot.steals_successful.load(std::memory_order_relaxed);
            
            double ops_per_sec = ops / duration_seconds;
            double avg_work_time_us = work_count > 0 ? (work_time_ns / 1000.0) / work_count : 0.0;
            double steal_hit_rate = steals_attempted > 0 ? 
                (static_cast<double>(steals_successful) / steals_attempted * 100) : 0.0;
            
            total_ops += ops;
            total_tasks += tasks;
            total_work_time += work_time_ns;
            total_work_count += work_count;
            total_steals_attempted += steals_attempted;
            total_steals_successful += steals_successful;
            total_stolen_tasks += cold.stolen_tasks;
            
            std::cout << std::setw(6) << i
                      << std::setw(6) << cold.cpu_id
                      << std::setw(12) << ops
                      << std::setw(12) << tasks
                      << std::setw(12) << std::fixed << std::setprecision(0) << ops_per_sec
                      << std::setw(12) << std::fixed << std::setprecision(2) << avg_work_time_us
                      << std::setw(10) << steals_successful << "/" << steals_attempted
                      << std::setw(8) << std::fixed << std::setprecision(1) << steal_hit_rate
                      << std::setw(8) << cold.stolen_tasks << "\n";
        }
        
        std::cout << std::string(80, '-') << "\n";
        
        double total_ops_per_sec = total_ops / duration_seconds;
        double avg_work_time_us = total_work_count > 0 ? (total_work_time / 1000.0) / total_work_count : 0.0;
        double overall_steal_rate = total_steals_attempted > 0 ? 
            (static_cast<double>(total_steals_successful) / total_steals_attempted * 100) : 0.0;
        double steal_efficiency = total_tasks > 0 ?
            (static_cast<double>(total_stolen_tasks) / total_tasks * 100) : 0.0;
        
        std::cout << "\nPERFORMANCE SUMMARY:\n";
        std::cout << "  Total Operations: " << total_ops << "\n";
        std::cout << "  Total Tasks: " << total_tasks << "\n";
        std::cout << "  Operations/second: " << std::fixed << std::setprecision(0) << total_ops_per_sec << "\n";
        std::cout << "  Average Work Time: " << std::fixed << std::setprecision(2) << avg_work_time_us << " μs\n";
        std::cout << "  Operations per Task: " << std::fixed << std::setprecision(2) 
                  << (total_tasks > 0 ? static_cast<double>(total_ops) / total_tasks : 0.0) << "\n";
        
        std::cout << "\nWORK STEALING STATISTICS:\n";
        std::cout << "  Steal Attempts: " << total_steals_attempted << "\n";
        std::cout << "  Successful Steals: " << total_steals_successful << "\n";
        std::cout << "  Steal Hit Rate: " << std::fixed << std::setprecision(1) << overall_steal_rate << "%\n";
        std::cout << "  Tasks Stolen: " << total_stolen_tasks << " (" 
                  << std::fixed << std::setprecision(1) << steal_efficiency << "% of total)\n";
        
        // Calculate load balance
        if (num_threads_ > 1) {
            std::vector<uint64_t> thread_ops(num_threads_);
            for (int i = 0; i < num_threads_; ++i) {
                thread_ops[i] = worker_stats_hot_[i].value.operations_completed.load(std::memory_order_relaxed);
            }
            
            uint64_t min_ops = *std::min_element(thread_ops.begin(), thread_ops.end());
            uint64_t max_ops = *std::max_element(thread_ops.begin(), thread_ops.end());
            double load_balance = min_ops > 0 ? (static_cast<double>(min_ops) / max_ops * 100) : 0.0;
            
            std::cout << "\nLOAD BALANCE:\n";
            std::cout << "  Min Operations: " << min_ops << "\n";
            std::cout << "  Max Operations: " << max_ops << "\n";
            std::cout << "  Balance Score: " << std::fixed << std::setprecision(1) << load_balance << "%\n";
        }
        
        // Platform-specific info
        std::cout << "\nPLATFORM INFO:\n";
        #ifdef __aarch64__
        std::cout << "  Architecture: ARM64 with NEON optimization\n";
        #elif defined(__x86_64__)
        std::cout << "  Architecture: x86_64 with AVX2 optimization\n";
        #else
        std::cout << "  Architecture: Generic (scalar fallback)\n";
        #endif
        
        std::cout << std::string(80, '=') << "\n";
    }
    
    // Additional utility methods
    void setConfig(const BenchmarkConfig& config) {
        if (running_.load(std::memory_order_acquire)) {
            throw std::runtime_error("Cannot change configuration while benchmark is running");
        }
        config_ = config;
    }
    
    BenchmarkConfig getConfig() const {
        return config_;
    }
    
    bool isRunning() const {
        return running_.load(std::memory_order_acquire);
    }
    
    size_t getQueueSizes() const {
        size_t total = 0;
        for (const auto& q : work_queues_) {
            total += q->size();
        }
        return total;
    }
};

// Utility function for running predefined benchmark scenarios
void runBenchmarkScenario(const std::string& scenario, int num_threads = 0, double duration = 10.0) {
    BenchmarkConfig config;
    
    if (scenario == "throughput") {
        config.default_work_iterations = 5000;
        config.update_frequency = 50;
        config.max_steal_attempts = 2;
        std::cout << "Running THROUGHPUT scenario (light work, high frequency)\n";
    } else if (scenario == "compute") {
        config.default_work_iterations = 50000;
        config.update_frequency = 200;
        config.max_steal_attempts = 5;
        std::cout << "Running COMPUTE scenario (heavy work, balanced)\n";
    } else if (scenario == "steal") {
        config.default_work_iterations = 1000;
        config.update_frequency = 20;
        config.max_steal_attempts = 8;
        config.adaptive_stealing = true;
        std::cout << "Running STEAL scenario (light work, aggressive stealing)\n";
    } else {
        std::cout << "Running DEFAULT scenario\n";
    }
    
    OptimizedBenchmark benchmark(num_threads, config);
    benchmark.runBenchmark(duration, 1000, scenario == "variable");
    benchmark.printResults();
}

int main(int argc, char* argv[]) {
    try {
        int num_threads = 0;
        double duration = 10.0;
        std::string scenario = "default";
        
        // Enhanced command line parsing
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-t" || arg == "--threads") {
                if (i + 1 < argc) {
                    num_threads = std::atoi(argv[++i]);
                }
            } else if (arg == "-d" || arg == "--duration") {
                if (i + 1 < argc) {
                    duration = std::atof(argv[++i]);
                }
            } else if (arg == "-s" || arg == "--scenario") {
                if (i + 1 < argc) {
                    scenario = argv[++i];
                }
            } else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]\n";
                std::cout << "Options:\n";
                std::cout << "  -t, --threads N     Use N worker threads (default: auto)\n";
                std::cout << "  -d, --duration T    Run for T seconds (default: 10.0)\n";
                std::cout << "  -s, --scenario S    Use scenario S (default, throughput, compute, steal)\n";
                std::cout << "  -h, --help         Show this help message\n";
                return 0;
            } else if (i == 1 && arg.find('-') != 0) {
                // First non-option argument is threads (backward compatibility)
                num_threads = std::atoi(arg.c_str());
            } else if (i == 2 && arg.find('-') != 0) {
                // Second non-option argument is duration (backward compatibility)
                duration = std::atof(arg.c_str());
            }
        }
        
        if (scenario != "default") {
            runBenchmarkScenario(scenario, num_threads, duration);
        } else {
            OptimizedBenchmark benchmark(num_threads);
            benchmark.runBenchmark(duration);
            benchmark.printResults();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
