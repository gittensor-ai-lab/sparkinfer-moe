#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace blackwell { namespace moe {

struct MoEConfig {
    int num_experts;
    int top_k;
    int hidden_dim;
    int ffn_dim;
    int num_layers;

    // Expert residency budget: how many expert weight matrices to keep in VRAM.
    // Remaining experts are evicted to CPU unified memory and fetched on demand.
    int expert_cache_slots;   // default: num_experts (all in VRAM)

    // Prefetch next-layer experts while current layer executes
    bool async_expert_prefetch = true;

    // Normalize routing weights after top-k selection
    bool normalize_expert_weights = true;
};

struct RoutingOutput {
    // expert_ids[token][k] = which expert processes token at rank k
    std::vector<std::vector<int>> expert_ids;
    // expert_weights[token][k] = weight for that expert
    std::vector<std::vector<float>> expert_weights;
};

class MoEEngine {
public:
    static std::unique_ptr<MoEEngine> create(const MoEConfig& cfg);
    virtual ~MoEEngine() = default;

    // Load expert weights for a layer. weights: [num_experts, hidden_dim, ffn_dim]
    virtual void load_layer_weights(int layer_idx, const void* weights) = 0;

    // Route tokens and dispatch through expert FFNs.
    // input:   [num_tokens, hidden_dim]  (fp16/bf16, device ptr)
    // output:  [num_tokens, hidden_dim]
    virtual void forward(
        const void* input, void* output,
        int num_tokens, int layer_idx,
        cudaStream_t stream
    ) = 0;

    // Expert cache management
    virtual void prefetch_experts(int layer_idx, const std::vector<int>& expert_ids) = 0;
    virtual void evict_experts(int layer_idx, const std::vector<int>& expert_ids) = 0;

    virtual const MoEConfig& config() const = 0;
};

}} // namespace blackwell::moe
