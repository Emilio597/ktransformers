/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_MOE_H
#define CPUINFER_OPERATOR_MOE_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <liburing.h>

#include "../../cpu_backend/backend.h"
#include "../../cpu_backend/shared_mem_buffer.h"
#include "conversion.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

struct SSDStreamConfig {
    std::string slice_dir; // 切片根目录，例如："/data/qwen3_slices/Lxx" 或单层路径
    int buffers_per_thread = 2; // 1 或 2（建议 2 为双缓冲）
    bool enable = false; // 开关
    // enum Layout { SeparateFiles = 0, PackedPerType = 1 } layout = SeparateFiles;
    int layer_id = -1;
    size_t align_bytes = 4096; // PackedPerType 对齐粒度
};

enum class ProjType : uint8_t { GATE = 0, UP = 1, DOWN = 2 };

// 切片请求键
struct SliceKey {
    ProjType type;
    int expert; // expert_id
    int expert_idx; // expert_id
    int ith; // slice id（行块）
};

// 切片尺寸信息（由 MOEConfig 推导）
struct SliceShape {
    size_t bytes_gate;
    size_t bytes_up;
    size_t bytes_down;
    int slices_gate_up; // = intermediate_size / stride
    int slices_down; // = hidden_size / stride
};


// ============ 文件路径/偏移计算 ============
struct LayoutHelper {
    SSDStreamConfig* cfg;
    SliceShape shape;

    static std::string proj_name(ProjType t){
        switch(t){
            case ProjType::GATE: return "gate";
            case ProjType::UP:   return "up";
            default:             return "down";
        }
    }

    size_t slice_bytes(ProjType t) const {
        // return (t==ProjType::GATE) ? shape.bytes_gate : (t==ProjType::UP ? shape.bytes_up : shape.bytes_down);
        size_t raw = (t==ProjType::GATE) ? shape.bytes_gate
                : (t==ProjType::UP)   ? shape.bytes_up
                                : shape.bytes_down;
        return align_up(raw, 4096); // 向上对齐到 4KB
    }

    // PackedPerType：每层每种类型 1 个文件，4KB 对齐
    std::string packed_path(ProjType t) const {
        char buf[512];
        ::snprintf(buf, sizeof(buf), "%s/%s.pack", cfg->slice_dir.c_str(), proj_name(t).c_str());
        return std::string(buf);
    }

    // (expert, ith) 在线性空间的位置
    size_t linear_index(const SliceKey& k) const {
        int spp = (k.type==ProjType::DOWN)? shape.slices_down : shape.slices_gate_up;
        return static_cast<size_t>(k.expert) * spp + k.ith;
    }

    // 用于 inflight 唯一 ID
    size_t inflight_index(const SliceKey& k) const {
        return ((size_t)k.expert << 32) |  (((size_t)k.type + 1) << 24)  | (size_t)k.ith; //防止为0
    }

    size_t align_up(size_t x, size_t a) const { return (x + a - 1) / a * a; }

    off_t packed_offset(const SliceKey& k) const {
        size_t slot = linear_index(k);
        size_t unit = align_up(slice_bytes(k.type), cfg->align_bytes);
        return static_cast<off_t>(slot * unit);
    }

    // 返回 (fd, offset) 与长度
    struct IOPlan { int fd; off_t off; size_t len; std::string owned_path; };

    IOPlan plan(const SliceKey& k) const {
        IOPlan p{ -1, 0, slice_bytes(k.type), {} };
        p.owned_path = packed_path(k.type);
        p.off = packed_offset(k);
        return p;
    }
};


// ============ SliceStreamer：每线程双缓冲 & 异步 I/O ============
class SliceStreamer {
public:
    struct Buffer {
        std::unique_ptr<uint8_t, void(*)(void*)> data{nullptr, ::operator delete};
        size_t cap = 0;
        std::atomic<bool> in_use{false};
        std::atomic<bool> ready{false};
        SliceKey key{};
    };
    struct ReadyTaskPointers {
        std::unordered_map<ProjType, uint8_t*> ptrs;
        SliceKey key; // 返回其中一个key作为代表即可
    };

    struct ThreadCtx {
        std::vector<std::unique_ptr<Buffer>> bufs;  // N 个缓冲（1 或 2）
        io_uring ring{};            // 每线程 1 个队列
        bool ring_inited = false;
        std::vector<size_t> in_flight;
        // std::unordered_map<std::string,int> fd_cache; // 路径 -> fd
        std::mutex mtx;
    };

    SliceStreamer(const SliceShape& shape, int threads, int buffers_per_thread)
        : threads_(threads), per_thread_(buffers_per_thread) {
        max_slice_ = std::max({shape.bytes_gate, shape.bytes_up, shape.bytes_down});
        ctxs_.resize(threads_);
        for (int t = 0; t < threads_; ++t) {
            ctxs_[t] = std::make_unique<ThreadCtx>();
            auto &ctx = *ctxs_[t];
            ctx.bufs.resize(per_thread_);
            ctx.in_flight.resize(per_thread_, 0);
            for (auto &b : ctx.bufs) {
                b = std::make_unique<Buffer>();
                void* ptr = ::operator new(max_slice_, std::align_val_t(4096));
                b->data = {reinterpret_cast<uint8_t*>(ptr), [](void* p){ ::operator delete(p, std::align_val_t(4096)); }};
                b->cap = max_slice_;
            }
            if (io_uring_queue_init(256, &ctx.ring, 0) == 0) ctx.ring_inited = true;
        }
    }

    ~SliceStreamer(){
        for (auto &ctx_ptr : ctxs_) {
            if (ctx_ptr->ring_inited) io_uring_queue_exit(&ctx_ptr->ring);
        }
    }

    int thread_index() const{  
        // 使用Backend的thread_local_id，它在do_work_stealing_job中被正确设置  
        int tid = Backend::thread_local_id;  
        if (tid < 0 || tid >= threads_) {  
            throw std::runtime_error("error thread id:"+ std::to_string(tid));  
        }  
        return tid;  
    }

    bool has_inflight() const {  
        int tid = thread_index();  
        return has_inflight_for_thread(tid);  
    }
    bool has_inflight_for_thread(int tid) const {  
        auto &ctx = *ctxs_[tid];  
        bool has = false;
        for (auto li : ctx.in_flight) {
            if (li != 0) { has = true; break; }
        }
        return has;
    }

    int inflight_size() const{
        int tid = thread_index(); 
        auto &ctx = *ctxs_[tid];  
        int has = 0;
        for (auto li : ctx.in_flight) {
            if (li != 0) has++; 
        }
        return has;
    }


    std::optional<ReadyTaskPointers> poll_if_all_ready(const LayoutHelper& lh_) {
        int tid = thread_index();
        return poll_if_all_ready_for_thread(tid, lh_);
    }

    std::optional<ReadyTaskPointers> poll_if_all_ready_for_thread(int tid, const LayoutHelper& lh_) {
        auto &ctx = *ctxs_[tid];

        int total_req = 0;
        for (auto li : ctx.in_flight) {
            if (li != 0) total_req++;
        }
        if (total_req == 0) return std::nullopt;

        // 1. 驱动IO完成队列
        if (ctx.ring_inited) {
            io_uring_cqe* cqe = nullptr;
            // 使用标准的 peek-seen 循环处理所有已完成的事件
            while (io_uring_peek_cqe(&ctx.ring, &cqe) == 0) {
                Buffer* b = reinterpret_cast<Buffer*>(io_uring_cqe_get_data(cqe));
                if (b) {
                    if (cqe->res >= 0) {
                        b->ready.store(true, std::memory_order_release);
                    } else {
                        fprintf(stderr, "io_uring read failed for expert %d slice %d with code: %d\n", 
                                b->key.expert, b->key.ith, cqe->res);
                        // IO失败，释放buffer并从inflight移除
                        b->in_use.store(false, std::memory_order_release);
                        b->ready.store(false, std::memory_order_release);
                        size_t li = lh_.inflight_index(b->key);
                        for (auto &slot : ctx.in_flight) {
                            if (slot == li) {
                                slot = 0; // 清除占用
                                break;
                            }
                        }
                    }
                }
                // 消费(acknowledges)这个完成事件
                io_uring_cqe_seen(&ctx.ring, cqe);
            }
        }

        int ready_count = 0;
        for (const auto& b : ctx.bufs) {
            if (b->in_use.load(std::memory_order_acquire) && b->ready.load(std::memory_order_acquire)) {
                ready_count++;
            }
        }

        if (ready_count > 0 && ready_count == total_req) {
            ReadyTaskPointers result;
            for (const auto& b : ctx.bufs) {
                if (b->in_use.load(std::memory_order_acquire)) {
                    result.ptrs[b->key.type] = b->data.get();
                    result.key = b->key;
                    b->ready.store(false, std::memory_order_relaxed);
                }
            }
            return result;
        }
        return std::nullopt;
    }

    SliceKey cancel_inflight_for_thread(int tid) {  
        SliceKey cancelled_key;  
        if (tid >= ctxs_.size()) 
            throw std::runtime_error("cancel_inflight error thread id:"+ std::to_string(tid));  
        auto &ctx = *ctxs_[tid];  
        
        cancelled_key = ctx.bufs[0]->key;
        
        if (ctx.ring_inited) {  
            // 使用 io_uring_prep_cancel 取消特定的 I/O 操作  
            for (int buf_idx = 0; buf_idx < (int)ctx.bufs.size(); ++buf_idx) {  
                if (ctx.in_flight[buf_idx] != 0) { // 有在途请求  
                    auto &buffer = ctx.bufs[buf_idx];  

                    io_uring_sqe* cancel_sqe = io_uring_get_sqe(&ctx.ring);  
                    if (cancel_sqe) {  
                        io_uring_prep_cancel(cancel_sqe, buffer.get(), 0);  
                        io_uring_sqe_set_data(cancel_sqe, nullptr);  
                    }  
                }  
            }  
            
            // 提交取消请求  
            io_uring_submit(&ctx.ring);  
            
            // 等待取消操作完成  
            io_uring_cqe* cqe = nullptr;  
            unsigned head;  
            io_uring_for_each_cqe(&ctx.ring, head, cqe) {  
                // 处理取消操作的完成事件  
                io_uring_cqe_seen(&ctx.ring, cqe);  
            }  
        }  
        
        // 重置所有该线程的 buffer 状态  
        for (auto &b : ctx.bufs) {  
            b->in_use.store(false, std::memory_order_release);  
            b->ready.store(false, std::memory_order_release);  
        }  
        
        // 清空该线程的 inflight 映射  
        std::fill(ctx.in_flight.begin(), ctx.in_flight.end(), 0); 
        
        return cancelled_key;  
    }

    // 已重命名
    void mark_consumed_all() {
        int tid = thread_index();
        auto &ctx = *ctxs_[tid];
        
        for (auto& b : ctx.bufs) {
            if (b->in_use.load(std::memory_order_acquire)) {
                b->in_use.store(false, std::memory_order_release);
            }
        }
        std::fill(ctx.in_flight.begin(), ctx.in_flight.end(), 0);
    }

    void mark_consumed_all_for_thread(int tid) {
        auto &ctx = *ctxs_[tid];
        
        for (auto& b : ctx.bufs) {
            if (b->in_use.load(std::memory_order_acquire)) {
                b->in_use.store(false, std::memory_order_release);
            }
        }
        std::fill(ctx.in_flight.begin(), ctx.in_flight.end(), 0);
    }

    // 查询是否有已就绪的切片；若有，返回 (buffer 指针, key)
    std::optional<std::pair<uint8_t*, SliceKey>> poll_ready(){
        int tid = thread_index();
        auto &ctx = *ctxs_[tid];

        // 驱动 completion（非阻塞）
        if (ctx.ring_inited) {
            io_uring_cqe* cqe = nullptr;
            unsigned head;
            io_uring_for_each_cqe(&ctx.ring, head, cqe) {
                Buffer* b = reinterpret_cast<Buffer*>(io_uring_cqe_get_data(cqe));
                if (cqe->res >= 0) {
                    b->ready.store(true, std::memory_order_release);
                } else {
                    // 读失败：降级为未就绪以便重试
                    fprintf(stderr, "io_uring read failed for expert %d slice %d with code: %d\n", 
                                b->key.expert, b->key.ith, cqe->res);
                    b->in_use.store(false, std::memory_order_release);
                }
                io_uring_cqe_seen(&ctx.ring, cqe);
            }
        }
        for (auto &b : ctx.bufs) if (b->ready.load()) {
                b->ready.store(false); b->in_use.store(false); 
                return std::make_optional(std::make_pair(b->data.get(), b->key));
            }
        return std::nullopt;
    }

    // 若该切片未在途，则占用一个空闲 buffer 并下发读取；返回是否成功发起
    bool ensure_inflight(const SliceKey& key,const LayoutHelper& lh_){
        int tid = thread_index();
        auto &ctx = *ctxs_[tid];
        size_t li = lh_.inflight_index(key);
        for (size_t infl : ctx.in_flight) {
            if (infl == li) return true; // 已在途
        }
        // 找空闲 buffer
        for (int i = 0; i < per_thread_; ++i) {
            auto &b = ctx.bufs[i];
            bool expect = false;
            if (b->in_use.compare_exchange_strong(expect, true)) {
                b->ready.store(false, std::memory_order_relaxed);
                b->key = key;
                // 打开文件 & 计算偏移
                auto plan = lh_.plan(key);
                int fd = get_fd(lh_.cfg->layer_id, key.type);  
                if (fd < 0) { b->in_use.store(false); return false; }
                if (ctx.ring_inited) {
                    io_uring_sqe* sqe = io_uring_get_sqe(&ctx.ring);
                    if (!sqe) { b->in_use.store(false); return false; }
                    io_uring_prep_read(sqe, fd, b->data.get(), (unsigned)plan.len, plan.off);
                    io_uring_sqe_set_data(sqe, b.get());
                    io_uring_submit(&ctx.ring);
                }
                ctx.in_flight[i] = li;
                return true;
            }
        }
        return false; // 无可用 buffer
    }
    // 在 SliceStreamer 类中新增  
    static void initialize_fd_cache(int max_layers) {  
        // std::lock_guard<std::mutex> lock(fd_cache_mutex_);  
        if (!fd_cache_initialized_) {  
            shared_fd_cache_.resize(max_layers);  
            for (auto& layer_fds : shared_fd_cache_) {  
                layer_fds.resize(3, -1); // 3个投影类型：GATE, UP, DOWN  
            }  
            fd_cache_initialized_ = true;  
        }  
    }  
    
    static void open_layer_files(int layer_id, const std::string& slice_dir) {  
        // std::lock_guard<std::mutex> lock(fd_cache_mutex_);  
        if (layer_id >= shared_fd_cache_.size()) {  
            shared_fd_cache_.resize(layer_id + 1);  
            for (int i = shared_fd_cache_.size() - (layer_id + 1 - shared_fd_cache_.size());   
                i <= layer_id; ++i) {  
                shared_fd_cache_[i].resize(3, -1);  
            }  
        }  
        
        // 打开三个投影文件  
        std::vector<std::string> proj_names = {"gate", "up", "down"};  
        for (int proj_idx = 0; proj_idx < 3; ++proj_idx) {  
            if (shared_fd_cache_[layer_id][proj_idx] == -1) {  
                char path_buf[512];  
                ::snprintf(path_buf, sizeof(path_buf), "%s/%s.pack",   
                        slice_dir.c_str(), proj_names[proj_idx].c_str());  
                
                int flags = O_RDONLY | O_CLOEXEC;  
                flags |= O_DIRECT;  
                int fd = ::open(path_buf, flags);  
                if (fd < 0) {  
                    fprintf(stderr, "Failed to open file '%s': %s (errno=%d)\\n",  
                            path_buf, strerror(errno), errno);  
                    std::exit(EXIT_FAILURE); 
                }  
                shared_fd_cache_[layer_id][proj_idx] = fd;  
            }  
        }  
    }  
    
    static void cleanup_shared_resources() {  
        // std::lock_guard<std::mutex> lock(SliceStreamer::fd_cache_mutex_);  
        for (auto& layer_fds : SliceStreamer::shared_fd_cache_) {  
            for (int fd : layer_fds) {  
                if (fd >= 0) ::close(fd);  
            }  
        }  
        SliceStreamer::shared_fd_cache_.clear();  
        SliceStreamer::fd_cache_initialized_ = false;  
    }

    static int get_fd(int layer_id, ProjType proj_type) {  
        // std::lock_guard<std::mutex> lock(fd_cache_mutex_);  
        if (layer_id >= shared_fd_cache_.size() ||   
            shared_fd_cache_[layer_id][(int)proj_type] == -1) {  
            return -1;  
        }  
        return shared_fd_cache_[layer_id][(int)proj_type];  
    }

private:
    // LayoutHelper lh_;
    static std::vector<std::vector<int>> shared_fd_cache_;  
    // static std::mutex fd_cache_mutex_;  
    static bool fd_cache_initialized_;  
    int threads_ = 1;
    int per_thread_ = 2;
    size_t max_slice_ = 0;
    std::vector<std::unique_ptr<ThreadCtx>> ctxs_;
};

struct MOEConfig {
    int expert_num;
    int routed_expert_num;
    int hidden_size;
    int intermediate_size;
    int stride;
    int group_min_len;
    int group_max_len;
    bool use_silu;
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type up_type;
    ggml_type down_type;
    ggml_type hidden_type;

    MOEConfig() {}

    MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, bool use_silu, void* gate_proj, void* up_proj, void* down_proj, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type)
        : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size), intermediate_size(intermediate_size), stride(stride), group_min_len(group_min_len), group_max_len(group_max_len), use_silu(use_silu), gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {}
};

class MOE {
   public:
    MOE(MOEConfig);
    ~MOE();
    void warm_up(Backend* backend);
    void forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend);
    void forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend);
    void forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, int* batch_size_tensor, Backend* backend);
    void enable_slice_compute(const std::string& slice_dir, int layer_id);
    void save_weight_slices(const std::string& output_dir);  
    void save_weight_slices_packed(const std::string& output_dir, int nth_gate_up, int nth_down);  
    void save_packed_projection(const std::string& output_dir, const std::string& proj_name,   
                            ProjType proj_type, int slices_per_expert,   
                            size_t slice_size, size_t unit_size);  
    size_t align_up(size_t x, size_t a) const;  
    void save_metadata(const std::string& output_dir);

    
   private:
    MOEConfig config_;
    void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

    #ifdef USE_NUMA
    std::vector<void*> gate_proj_numa_;  // [numa_num, expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    std::vector<void*> up_proj_numa_;    // [numa_num, expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    std::vector<void*> down_proj_numa_;  // [numa_num, expert_num * hidden_size * intermediate_size ( /32 if quantized)]
    #endif

    float* s_input_fp32_;                      // [hidden_size]
    uint8_t* s_gate_input_;                    // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* s_up_input_;                      // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    std::vector<float*> s_gate_output_;        // [routed_expert_num, intermediate_size]
    std::vector<float*> s_up_output_;          // [routed_expert_num, intermediate_size]
    std::vector<float*> s_intermediate_fp32_;  // [routed_expert_num, intermediate_size]
    std::vector<uint8_t*> s_down_input_;       // [routed_expert_num, intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    std::vector<float*> s_down_output_;        // [routed_expert_num, hidden_size]
    float* s_output_fp32_;                     // [hidden_size]

    std::vector<float*> m_input_fp32_;    // [group_max_len, hidden_size]
    std::vector<uint8_t*> m_gate_input_;  // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    std::vector<uint8_t*> m_up_input_;    // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    uint8_t* m_local_gate_input_;         // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* m_local_up_input_;           // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    float* m_local_gate_output_;          // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_up_output_;            // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_intermediate_fp32_;    // [routed_expert_num * group_max_len * intermediate_size]
    uint8_t* m_local_down_input_;         // [routed_expert_num * group_max_len * intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    float* m_local_down_output_;          // [routed_expert_num * group_max_len * hidden_size]
    std::vector<float*> m_output_fp32_;   // [group_max_len, hidden_size]

    std::vector<std::vector<int>> m_local_pos_;          // [group_max_len, routed_expert_num]
    std::vector<int> m_local_num_;                       // [expert_num]
    std::vector<uint8_t*> m_local_gate_input_ptr_;       // [expert_num]
    std::vector<uint8_t*> m_local_up_input_ptr_;         // [expert_num]
    std::vector<float*> m_local_gate_output_ptr_;        // [expert_num]
    std::vector<float*> m_local_up_output_ptr_;          // [expert_num]
    std::vector<float*> m_local_intermediate_fp32_ptr_;  // [expert_num]
    std::vector<uint8_t*> m_local_down_input_ptr_;       // [expert_num]
    std::vector<float*> m_local_down_output_ptr_;        // [expert_num]

    static std::unique_ptr<SliceStreamer> streamer_;  
    static std::once_flag streamer_init_flag_;  
    // static LayoutHelper layout_helper_;  
    static SliceShape slice_shape_;  
    static int worker_threads_;  

    SSDStreamConfig ssd_cfg_;
    LayoutHelper layout_helper_;
    // std::unique_ptr<SliceStreamer> streamer_;
    // SliceShape slice_shape_;
    // int worker_threads_ = std::max(1u, std::thread::hardware_concurrency());
};

#endif