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
#ifdef USE_IO_URING
#include <liburing.h>
#endif

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
        return (t==ProjType::GATE) ? shape.bytes_gate : (t==ProjType::UP ? shape.bytes_up : shape.bytes_down);
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

//     // 查询是否有已就绪的切片；若有，返回 (buffer 指针, key)
//     std::optional<std::pair<uint8_t*, SliceKey>> poll_ready(){
//         int tid = thread_index();
//         auto &ctx = *ctxs_[tid];
//         for (auto &b : ctx.bufs) {
//             if (b->ready.load(std::memory_order_acquire)) {
//                 b->ready.store(false, std::memory_order_release);
//                 b->in_use.store(false, std::memory_order_release);
//                 return std::make_optional(std::make_pair(b->data.get(), b->key));
//             }
//         }
// #ifdef USE_IO_URING
//         // 驱动 completion（非阻塞）
//         if (ctx.ring_inited) {
//             io_uring_cqe* cqe = nullptr;
//             unsigned head;
//             io_uring_for_each_cqe(&ctx.ring, head, cqe) {
//                 Buffer* b = reinterpret_cast<Buffer*>(io_uring_cqe_get_data(cqe));
//                 if (cqe->res >= 0) {
//                     b->ready.store(true, std::memory_order_release);
//                 } else {
//                     // 读失败：降级为未就绪以便重试
//                     b->in_use.store(false, std::memory_order_release);
//                 }
//                 io_uring_cqe_seen(&ctx.ring, cqe);
//             }
//             for (auto &b : ctx.bufs) if (b->ready.load()) {
//                 b->ready.store(false); b->in_use.store(false); return std::make_optional(std::make_pair(b->data.get(), b->key));
//             }
//         }
// #endif
//         return std::nullopt;
//     }


// ============ SliceStreamer：每线程双缓冲 & 异步 I/O ============
class SliceStreamer {
public:
    struct Buffer {
        std::unique_ptr<uint8_t[]> data;
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
#ifdef USE_IO_URING
        io_uring ring{};            // 每线程 1 个队列
        bool ring_inited = false;
#endif
        std::unordered_map<size_t, int> in_flight; // linear_index -> buffer idx
        std::unordered_map<std::string,int> fd_cache; // 路径 -> fd
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
            for (auto &b : ctx.bufs) {
                b = std::make_unique<Buffer>();
                b->data.reset(new uint8_t[max_slice_]);
                b->cap = max_slice_;
            }
        #ifdef USE_IO_URING
            if (io_uring_queue_init(256, &ctx.ring, 0) == 0) ctx.ring_inited = true;
        #endif
        }
    }

    ~SliceStreamer(){
    #ifdef USE_IO_URING
        for (auto &ctx_ptr : ctxs_) {
            if (ctx_ptr->ring_inited) io_uring_queue_exit(&ctx_ptr->ring);
        }
    #endif
        for (auto &ctx_ptr : ctxs_) {
            for (auto &kv : ctx_ptr->fd_cache) {
                if (kv.second >= 0) ::close(kv.second);
            }
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
        auto &ctx = *ctxs_[tid];  
        return !ctx.in_flight.empty();  
    }
    bool has_inflight_for_thread(int tid) const {  
        auto &ctx = *ctxs_[tid];  
        return !ctx.in_flight.empty();  
    }

    std::optional<ReadyTaskPointers> poll_if_all_ready(const LayoutHelper& lh_) {
        int tid = thread_index();
        return poll_if_all_ready_for_thread(tid, lh_);
    }

    std::optional<ReadyTaskPointers> poll_if_all_ready_for_thread(int tid, const LayoutHelper& lh_) {
        auto &ctx = *ctxs_[tid];

        if (ctx.in_flight.empty()) return std::nullopt;

    #ifdef USE_IO_URING
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
                        size_t li = lh_.linear_index(b->key);
                        ctx.in_flight.erase(li);
                    }
                }
                // 消费(acknowledges)这个完成事件
                io_uring_cqe_seen(&ctx.ring, cqe);
            }
        }
    #endif

        // 2. 检查是否所有在途(in_flight)的请求都已就绪
        // 注意：如果IO失败，in_flight条目会被移除，所以这个检查依然是正确的
        if (ctx.in_flight.empty()) return std::nullopt; // 可能所有请求都失败了

        int ready_count = 0;
        for (const auto& b : ctx.bufs) {
            if (b->in_use.load(std::memory_order_acquire) && b->ready.load(std::memory_order_acquire)) {
                ready_count++;
            }
        }

        if (ready_count > 0 && ready_count == ctx.in_flight.size()) {
            ReadyTaskPointers result;
            bool key_saved = false;
            for (const auto& b : ctx.bufs) {
                if (b->in_use.load(std::memory_order_acquire)) {
                    result.ptrs[b->key.type] = b->data.get();
                    if (!key_saved) {
                        result.key = b->key;
                        key_saved = true;
                    }
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
        
    #ifdef USE_IO_URING  
        if (ctx.ring_inited) {  
            // 使用 io_uring_prep_cancel 取消特定的 I/O 操作  
            for (const auto& [linear_idx, buffer_idx] : ctx.in_flight) {  
                if (buffer_idx < ctx.bufs.size()) {  
                    auto &buffer = ctx.bufs[buffer_idx];  
                    
                    // 准备取消操作  
                    io_uring_sqe* cancel_sqe = io_uring_get_sqe(&ctx.ring);  
                    if (cancel_sqe) {  
                        // 使用 buffer 地址作为 user_data 来标识要取消的操作  
                        io_uring_prep_cancel(cancel_sqe, buffer.get(), 0);  
                        io_uring_sqe_set_data(cancel_sqe, nullptr); // 取消操作不需要回调  
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
    #endif  
        
        // 重置所有该线程的 buffer 状态  
        for (auto &b : ctx.bufs) {  
            b->in_use.store(false, std::memory_order_release);  
            b->ready.store(false, std::memory_order_release);  
        }  
        
        // 清空该线程的 inflight 映射  
        ctx.in_flight.clear();  
        
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
        ctx.in_flight.clear();
    }

    void mark_consumed_all_for_thread(int tid) {
        auto &ctx = *ctxs_[tid];
        
        for (auto& b : ctx.bufs) {
            if (b->in_use.load(std::memory_order_acquire)) {
                b->in_use.store(false, std::memory_order_release);
            }
        }
        ctx.in_flight.clear();
    }

    // 若该切片未在途，则占用一个空闲 buffer 并下发读取；返回是否成功发起
    bool ensure_inflight(const SliceKey& key,const LayoutHelper& lh_){
        int tid = thread_index();
        auto &ctx = *ctxs_[tid];
        size_t li = lh_.linear_index(key);
        if (ctx.in_flight.count(li)) return true; // 已在途
        // 找空闲 buffer
        for (int i = 0; i < per_thread_; ++i) {
            auto &b = ctx.bufs[i];
            bool expect = false;
            if (b->in_use.compare_exchange_strong(expect, true)) {
                b->ready.store(false, std::memory_order_relaxed);
                b->key = key;
                // 打开文件 & 计算偏移
                auto plan = lh_.plan(key);
                int fd = -1;
                {
                    std::lock_guard<std::mutex> lk(ctx.mtx);
                    auto it = ctx.fd_cache.find(plan.owned_path);
                    if (it == ctx.fd_cache.end()) {
                        int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_DIRECT
                        flags |= O_DIRECT; // 提升吞吐（注意对齐要求）
#endif
                        fd = ::open(plan.owned_path.c_str(), flags);
                        ctx.fd_cache[plan.owned_path] = fd;
                    } else fd = it->second;
                }
                if (fd < 0) { b->in_use.store(false); return false; }
#ifdef USE_IO_URING
                if (ctx.ring_inited) {
                    io_uring_sqe* sqe = io_uring_get_sqe(&ctx.ring);
                    if (!sqe) { b->in_use.store(false); return false; }
                    io_uring_prep_read(sqe, fd, b->data.get(), (unsigned)plan.len, plan.off);
                    io_uring_sqe_set_data(sqe, b.get());
                    io_uring_submit(&ctx.ring);
                } else
#endif
                {
                    // 回退：同步 pread（仍然通过状态机）
                    ssize_t n = ::pread(fd, b->data.get(), plan.len, plan.off);
                    if (n == (ssize_t)plan.len) {
                        b->ready.store(true, std::memory_order_release);
                    } else {
                        b->in_use.store(false, std::memory_order_release);
                        return false;
                    }
                }
                ctx.in_flight[li] = i;
                return true;
            }
        }
        return false; // 无可用 buffer
    }

private:
    // LayoutHelper lh_;
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
    void enable_slice_compute(const std::string& slice_dir);
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