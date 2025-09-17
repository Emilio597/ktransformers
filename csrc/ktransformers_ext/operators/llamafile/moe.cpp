/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-15 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "moe.h"
#include <iostream>
#include <cstdint>
#include <math.h>
#include <climits>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

#define MAX_LAYER 100
// #define TIME_PERF
// #define JOB_DEBUG

std::unique_ptr<SliceStreamer> MOE::streamer_;  
std::once_flag MOE::streamer_init_flag_;  
// LayoutHelper MOE::layout_helper_;  
SliceShape MOE::slice_shape_;  
int MOE::worker_threads_;
// SSDStreamConfig MOE::ssd_cfg_;

std::vector<std::vector<int>> SliceStreamer::shared_fd_cache_;  
// std::mutex SliceStreamer::fd_cache_mutex_;  
bool SliceStreamer::fd_cache_initialized_ = false;

MOE::MOE(MOEConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;
    
    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    gate_proj_numa_.resize(numa_nodes);
    up_proj_numa_.resize(numa_nodes);
    down_proj_numa_.resize(numa_nodes);
    size_t exp_inter_hidden_mul_ = (size_t)config.expert_num * config.intermediate_size * config.hidden_size;
    for (int i = 0; i < numa_nodes; i++) {
        gate_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type), i);
        up_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type), i);
        down_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type), i);
        if (!gate_proj_numa_[i]) {
            std::cout << "Memory allocation failed for gate_proj_numa_ on node " << i << std::endl;
        }
        if (!up_proj_numa_[i]) {
            std::cout << "Memory allocation failed for up_proj_numa_ on node " << i << std::endl;
        }
        if (!down_proj_numa_[i]) {
            std::cout << "Memory allocation failed for down_proj_numa_ on node " << i << std::endl;
        }
        memcpy(gate_proj_numa_[i], gate_proj_, exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type));
        memcpy(up_proj_numa_[i], up_proj_, exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type));
        memcpy(down_proj_numa_[i], down_proj_, exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type));
    }
    #endif

    std::vector<std::pair<void**, uint64_t>> s_mem_requests;
    s_mem_requests.push_back({(void**)&s_input_fp32_, sizeof(float) * config_.hidden_size});
    s_mem_requests.push_back({(void**)&s_gate_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
    s_mem_requests.push_back({(void**)&s_up_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    s_gate_output_.resize(config_.routed_expert_num);
    s_up_output_.resize(config_.routed_expert_num);
    s_intermediate_fp32_.resize(config_.routed_expert_num);
    s_down_input_.resize(config_.routed_expert_num);
    s_down_output_.resize(config_.routed_expert_num);
    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_mem_requests.push_back({(void**)&s_gate_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_up_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_down_input_[i], config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type)});
        s_mem_requests.push_back({(void**)&s_down_output_[i], sizeof(float) * config_.hidden_size});
    }
    s_mem_requests.push_back({(void**)&s_output_fp32_, sizeof(float) * config_.hidden_size});
    shared_mem_buffer.alloc(this, s_mem_requests);

    std::vector<std::pair<void**, uint64_t>> m_mem_requests;
    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_input_fp32_[i], sizeof(float) * config_.hidden_size});
        m_mem_requests.push_back({(void**)&m_gate_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
        m_mem_requests.push_back({(void**)&m_up_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    }
    m_mem_requests.push_back({(void**)&m_local_gate_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_up_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_gate_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_up_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_intermediate_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_down_input_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_down_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_output_fp32_[i], sizeof(float) * config_.hidden_size});
    }
    shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_local_pos_[i].resize(config_.routed_expert_num);
    }
    m_local_num_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    std::call_once(streamer_init_flag_, [&]() {  
        slice_shape_.bytes_gate = (size_t)config_.stride * config_.hidden_size *  
            ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);  
        slice_shape_.bytes_up = (size_t)config_.stride * config_.hidden_size *  
            ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);  
        slice_shape_.bytes_down = (size_t)config_.stride * config_.intermediate_size *  
            ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);  
        slice_shape_.slices_gate_up = config_.intermediate_size / config_.stride;  
        slice_shape_.slices_down = config_.hidden_size / config_.stride;  
        worker_threads_ = std::max(1u, std::thread::hardware_concurrency());
        streamer_.reset(new SliceStreamer(slice_shape_, worker_threads_, ssd_cfg_.buffers_per_thread));  
        SliceStreamer::initialize_fd_cache(MAX_LAYER);//暂时固化，python端未传递层数
        std::cout << "SliceStreamer init worker threads: "<<worker_threads_ << std::endl;
    });  
    layout_helper_.cfg = &ssd_cfg_;  
    layout_helper_.shape = slice_shape_; 
    if (slice_shape_.slices_gate_up !=  config_.intermediate_size / config_.stride) {  
        throw std::runtime_error("MOE configuration mismatch with global SliceStreamer");  
    }
}

MOE::~MOE() {
    shared_mem_buffer.dealloc(this);

    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    for (int i = 0; i < numa_nodes; i++) {
        numa_free(gate_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type));
        numa_free(up_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type));
        numa_free(down_proj_numa_[i], config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type));
    }
    #endif
}

void MOE::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
    for (int i = 0; i < config_.expert_num; i++) {
        uint64_t expert_ids = i;
        float weights = 0;
        forward_one(1, &expert_ids, &weights, input.data(), output.data(), backend);
    }
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

static float act_fn_relu(float x) {
    if(x > 0.0){
        return x;
    } else {
        return 0.0;
    }
}
#ifdef TIME_PERF
std::mutex timing_mutex;
#endif
#ifdef JOB_DEBUG
std::mutex count_mutex;
#endif
void MOE::forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = s_gate_input_;
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = s_gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(s_input_fp32_, s_up_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = s_up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }
    int nth = config_.intermediate_size / config_.stride;
#ifdef TIME_PERF
    // 添加时间统计变量  
    long long gate_total = 0;
    long long gate_min = LLONG_MAX;
    long long gate_max = LLONG_MIN;
      
    // Down projection统计变量
    long long down_total = 0;
    long long down_min = LLONG_MAX;
    long long down_max = LLONG_MIN;
#endif
#ifdef JOB_DEBUG
    int first_count = 0;
    int ready_count = 0;
    int memory_count = 0;
    int timeout_count = 0;
#endif
    // remaining_tasks.store(nth * k, std::memory_order_relaxed);
    // int degrade_threshold = backend->get_iothread_num()/2;
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
#ifdef TIME_PERF
        auto start_time = std::chrono::high_resolution_clock::now();  
#endif
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;
        
        void* gate_proj_ptr = nullptr;
        void* up_proj_ptr   = nullptr;
        bool io_catch = false;
        ThreadType current_type = Backend::get_thread_type();  
        if (ssd_cfg_.enable && current_type == ThreadType::IO_THREAD) {
            // if (remaining_tasks.load(std::memory_order_acquire) < degrade_threshold) {
            //     goto MEMORY_FALLBACK;
            // }
            SliceKey kg{ProjType::GATE, (int)expert_id, expert_idx, ith};
            SliceKey ku{ProjType::UP, (int)expert_id, expert_idx, ith};
            if (!streamer_->has_inflight()) {
                // case 1: 没有 inflight（首次）
                if((!streamer_->ensure_inflight(kg, layout_helper_)) || (!streamer_->ensure_inflight(ku, layout_helper_)))
                {
                    std::exit(EXIT_FAILURE);
                }
#ifdef JOB_DEBUG
                {
                    std::lock_guard<std::mutex> lock(count_mutex);
                    // std::cout<<"[streamer] tid="<<streamer_->thread_index()<<" ensure_inflight" <<std::endl;
                    first_count++;
                }
#endif
                usleep(200);
            }
            else
                assert(0);
            int count=0;
            do{
                auto ready_pair = streamer_->poll_if_all_ready(layout_helper_);
                if (ready_pair) {
                    // case 2: inflight 有且 ready
                    auto p = ready_pair->ptrs;
                    // auto key = ready_pair->key;
                    gate_proj_ptr = p[ProjType::GATE];
                    up_proj_ptr = p[ProjType::UP];
                    io_catch = true;
                    // expert_id = key.expert;
                    // expert_idx = key.expert_idx;
                    // ith = key.ith;
    #ifdef JOB_DEBUG
                    {
                        std::lock_guard<std::mutex> lock(count_mutex);
                        ready_count++;
                    }
    #endif
                    break;
                }
                else
                {
                    count++;
                    if(count>3)
                    {
                        #ifdef JOB_DEBUG
                        {
                            std::lock_guard<std::mutex> lock(count_mutex);
                            timeout_count++;
                        }
                        #endif
                        streamer_->cancel_inflight();
                        goto MEMORY_FALLBACK;
                    }
                    // if (remaining_tasks.load(std::memory_order_acquire) < degrade_threshold) {
                    //     goto MEMORY_FALLBACK;
                    // }
                    // streamer_->cancel_inflight();
                    usleep(100);
                }
            }while(1);
        }
MEMORY_FALLBACK:
        if(!gate_proj_ptr){
            #ifdef USE_NUMA
            gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
            #else
            gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
            #endif

            #ifdef USE_NUMA
            up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
            #else
            up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
            #endif
#ifdef JOB_DEBUG
                {
                    std::lock_guard<std::mutex> lock(count_mutex);
                    memory_count++;
                }
#endif
        }

        float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

        float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        if(config_.use_silu){
            // use silu as act fn
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
            }
        } else {
            // use relu as act fn
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_intermediate_fp32_[expert_idx][i] = act_fn_relu(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
            }
        }
        if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.stride;
            void* down_input_ptr = s_down_input_[expert_idx] + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
        // remaining_tasks.fetch_sub(1, std::memory_order_acq_rel);
        if(io_catch)
            streamer_->mark_consumed_all();
        
#ifdef TIME_PERF
        auto end_time = std::chrono::high_resolution_clock::now();  
        auto t = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
         // 多线程安全更新
        {
            std::lock_guard<std::mutex> lock(timing_mutex);
            gate_total += t;
            if (t < gate_min) gate_min = t;
            if (t > gate_max) gate_max = t;
        }
#endif
    }, nullptr);

//     if (ssd_cfg_.enable) {  
//         backend->do_work_stealing_job(worker_threads_, nullptr, [&](int thread_id) {  
//             // 每个线程处理自己的 inflight I/O  
//             if(streamer_->has_inflight_for_thread(thread_id)) {  
//                 int expert_idx;
//                 uint64_t expert_id;
//                 int ith;
                
//                 void* gate_proj_ptr = nullptr;
//                 void* up_proj_ptr   = nullptr;
//                 auto ready_pair = streamer_->poll_if_all_ready_for_thread(thread_id, layout_helper_);  
//                 if (ready_pair) {  
//                     // 数据已就绪，标记为消费完成  
//                     auto p = ready_pair->ptrs;
//                     auto key = ready_pair->key;
//                     gate_proj_ptr = p[ProjType::GATE];
//                     up_proj_ptr = p[ProjType::UP];
//                     expert_id = key.expert;
//                     expert_idx = key.expert_idx;
//                     ith = key.ith;
//                     streamer_->mark_consumed_all_for_thread(thread_id);  
// #ifdef JOB_DEBUG
//                 {
//                     std::lock_guard<std::mutex> lock(count_mutex);
//                     assert(up_proj_ptr != nullptr && gate_proj_ptr!=nullptr);
//                     ready_count++;
//                 }
// #endif
//                 } else {  
//                     // 数据未就绪，取消所有该线程的 inflight I/O  
//                     auto key = streamer_->cancel_inflight_for_thread(thread_id);  
//                     expert_id = key.expert;
//                     expert_idx = key.expert_idx;
//                     ith = key.ith;                    
// #ifdef JOB_DEBUG
//                 {
//                     std::lock_guard<std::mutex> lock(count_mutex);
//                     memory_count++;
//                 }
// #endif
//                 }  
//                 if(!gate_proj_ptr){
//                     #ifdef USE_NUMA
//                     gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
//                     #else
//                     gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
//                     #endif
//                 }
//                 if(!up_proj_ptr){
//                     #ifdef USE_NUMA
//                     up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
//                     #else
//                     up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
//                     #endif
//                 }
//                 float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride;
//                 llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

//                 float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;
//                 llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
//                 if(config_.use_silu){
//                     // use silu as act fn
//                     for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
//                         s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
//                     }
//                 } else {
//                     // use relu as act fn
//                     for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
//                         s_intermediate_fp32_[expert_idx][i] = act_fn_relu(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
//                     }
//                 }
//                 if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
//                     float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.stride;
//                     void* down_input_ptr = s_down_input_[expert_idx] + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
//                     from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
//                 }
//             }  
//         }, nullptr);  
//     }
#ifdef TIME_PERF
    int gate_count = nth * k;
    double gate_avg = gate_count > 0 ? static_cast<double>(gate_total) / gate_count : 0.0;
    std::cout << "=== Forward One Timing Statistics ===" << std::endl;
    std::cout << "Gate+Up projection jobs:" << std::endl;
    std::cout << " Count: " << gate_count << std::endl;
    std::cout << " Avg: " << gate_avg << " μs" << std::endl;
    std::cout << " Min: " << gate_min << " μs" << std::endl;
    std::cout << " Max: " << gate_max << " μs" << std::endl;
#endif
#ifdef JOB_DEBUG
    std::cout << " first_count: " << first_count <<  std::endl;
    std::cout << " ready_count: " << ready_count <<  std::endl;
    std::cout << " memory_count: " << memory_count <<  std::endl;
    std::cout << " timeout_count: " << timeout_count <<  std::endl;
    if(memory_count + ready_count != gate_count)
        throw std::runtime_error("miss task found!"); 
#endif
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        for (int i = 0; i < k; i++) {
            from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
#ifdef TIME_PERF
        auto start_time = std::chrono::high_resolution_clock::now(); 
#endif
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            #ifdef USE_NUMA
            void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #else
            void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #endif
            
            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            float* output_fp32_ptr = s_output_fp32_ + ith * config_.stride;
            void* output_ptr = (uint8_t*)output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
#ifdef TIME_PERF
        auto end_time = std::chrono::high_resolution_clock::now(); 
        auto t = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        {
            std::lock_guard<std::mutex> lock(timing_mutex);
            down_total += t;
            if (t < down_min) down_min = t;
            if (t > down_max) down_max = t;
        }
#endif
    }, nullptr);
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        from_float(s_output_fp32_, output, config_.hidden_size, config_.hidden_type);
    }
    int down_count = nth;
#ifdef TIME_PERF
    double down_avg = down_count > 0 ? static_cast<double>(down_total) / down_count : 0.0;
    std::cout << "Down projection jobs:" << std::endl;
    std::cout << " Count: " << down_count << std::endl;
    std::cout << " Avg: " << down_avg << " μs" << std::endl;
    std::cout << " Min: " << down_min << " μs" << std::endl;
    std::cout << " Max: " << down_max << " μs" << std::endl;
#endif
}

void MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        }
    }, nullptr);
    int stride = QK_K;
    int nth = config_.intermediate_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            if(config_.use_silu){
                for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                    m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
                }
            } else {
                for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                    m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn_relu(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
                }
            }
            float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    stride = QK_K;
    nth = config_.hidden_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
        
        #ifdef USE_NUMA
        void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #else
        void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #endif

        float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * k + j];
            }
        }
        from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);
}

void MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, int* batch_size_tensor, Backend* backend) {
    qlen = batch_size_tensor[0];
    if (qlen < config_.group_min_len) {
        for (int i = 0; i < qlen; i++) {
            forward_one(k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
        }
        return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    forward_many(forward_len, k, expert_ids, weights, input, output, backend);

    batch_size_tensor[0] -= forward_len;
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k, (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), batch_size_tensor, backend);
}

void MOE::enable_slice_compute(const std::string& slice_dir, int layer_id) { 
    ssd_cfg_.slice_dir = slice_dir;
    ssd_cfg_.enable = true;
    
    ssd_cfg_.layer_id = layer_id; // 默认值  
    // 预先打开该层的所有文件  
    SliceStreamer::open_layer_files(ssd_cfg_.layer_id, slice_dir);  
}

void MOE::save_weight_slices(const std::string& output_dir) {  
    // 创建输出目录  
    
    std::filesystem::create_directories(output_dir);  
      
    int nth_gate_up = config_.intermediate_size / config_.stride;  
    int nth_down = config_.hidden_size / config_.stride;  
      
    // 固定使用PackedPerType布局  
    save_weight_slices_packed(output_dir, nth_gate_up, nth_down);  
      
    // 保存元数据文件  
    save_metadata(output_dir);  
}  
  
void MOE::save_weight_slices_packed(const std::string& output_dir, int nth_gate_up, int nth_down) {  
    const size_t align_bytes = 4096; // 4KB对齐  
      
    // 计算每种投影类型的切片大小  
    size_t gate_slice_size = config_.stride * config_.hidden_size *   
                            ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);  
    size_t up_slice_size = config_.stride * config_.hidden_size *   
                          ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);  
    size_t down_slice_size = config_.stride * config_.intermediate_size *   
                            ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);  
      
    // 计算对齐后的单元大小  
    size_t gate_unit_size = align_up(gate_slice_size, align_bytes);  
    size_t up_unit_size = align_up(up_slice_size, align_bytes);  
    size_t down_unit_size = align_up(down_slice_size, align_bytes);  
      
    // 保存gate投影打包文件  
    save_packed_projection(output_dir, "gate", ProjType::GATE, nth_gate_up,   
                          gate_slice_size, gate_unit_size);  
      
    // 保存up投影打包文件  
    save_packed_projection(output_dir, "up", ProjType::UP, nth_gate_up,   
                          up_slice_size, up_unit_size);  
      
    // 保存down投影打包文件  
    save_packed_projection(output_dir, "down", ProjType::DOWN, nth_down,   
                          down_slice_size, down_unit_size);  
}  
  
void MOE::save_packed_projection(const std::string& output_dir, const std::string& proj_name,   
                                ProjType proj_type, int slices_per_expert,   
                                size_t slice_size, size_t unit_size) {  
    std::string packed_filename = output_dir + "/" + proj_name + ".pack";  
    std::ofstream packed_file(packed_filename, std::ios::binary);  
    if (!packed_file) {  
        int err = errno;
        throw std::runtime_error("Failed to create packed file: " + packed_filename
                                + " errno=" + std::to_string(err) + " (" + std::string(std::strerror(err)) + ")");
    }  
      
    // 计算总文件大小  
    size_t total_slices = config_.expert_num * slices_per_expert;  
    size_t total_size = total_slices * unit_size;  
      
    // 创建对齐缓冲区  
    std::vector<uint8_t> aligned_buffer(unit_size, 0);  
      
    for (int expert_id = 0; expert_id < config_.expert_num; expert_id++) {  
        for (int ith = 0; ith < slices_per_expert; ith++) {  
            // 清零缓冲区  
            std::fill(aligned_buffer.begin(), aligned_buffer.end(), 0);  
              
            // 计算源数据偏移  
            size_t src_offset;  
            const uint8_t* src_ptr;  
              
            if (proj_type == ProjType::GATE) {  
                src_offset = (expert_id * config_.intermediate_size + ith * config_.stride) *  
                            config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);  
                src_ptr = (uint8_t*)gate_proj_ + src_offset;  
            } else if (proj_type == ProjType::UP) {  
                src_offset = (expert_id * config_.intermediate_size + ith * config_.stride) *  
                            config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);  
                src_ptr = (uint8_t*)up_proj_ + src_offset;  
            } else { // DOWN  
                src_offset = (expert_id * config_.hidden_size + ith * config_.stride) *  
                            config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);  
                src_ptr = (uint8_t*)down_proj_ + src_offset;  
            }  
              
            // 复制数据到对齐缓冲区  
            std::memcpy(aligned_buffer.data(), src_ptr, slice_size);  
              
            // 写入对齐的数据块  
            packed_file.write(reinterpret_cast<const char*>(aligned_buffer.data()), unit_size);  
            if (!packed_file) {  
                throw std::runtime_error("Failed to write to packed file: " + packed_filename);  
            }  
        }  
    }  
      
    packed_file.close();  
    std::cout << "Saved packed " << proj_name << " projection to " << packed_filename   
              << " (total size: " << total_size << " bytes)" << std::endl;  
}  
  
size_t MOE::align_up(size_t x, size_t a) const {  
    return (x + a - 1) / a * a;  
}  
  
void MOE::save_metadata(const std::string& output_dir) {  
    std::string metadata_file = output_dir + "/metadata.json";  
    std::ofstream file(metadata_file);  
    if (!file) {  
        throw std::runtime_error("Failed to create metadata file");  
    }  
  
    file << "{\n";  
    file << "  \"expert_num\": " << config_.expert_num << ",\n";  
    file << "  \"intermediate_size\": " << config_.intermediate_size << ",\n";  
    file << "  \"hidden_size\": " << config_.hidden_size << ",\n";  
    file << "  \"stride\": " << config_.stride << ",\n";  
    file << "  \"gate_type\": " << config_.gate_type << ",\n";  
    file << "  \"up_type\": " << config_.up_type << ",\n";  
    file << "  \"down_type\": " << config_.down_type << ",\n";  
    file << "  \"gate_slices_per_expert\": " << (config_.intermediate_size / config_.stride) << ",\n";  
    file << "  \"down_slices_per_expert\": " << (config_.hidden_size / config_.stride) << ",\n";  
    file << "  \"layout\": \"PackedPerType\",\n";  
    file << "  \"align_bytes\": 4096\n";  
    file << "}\n";  
  
    file.close();  
}