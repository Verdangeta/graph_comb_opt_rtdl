#include "nstep_replay_mem.h"
#include "i_env.h"
#include "config.h"
#include <cassert>
#include <algorithm>
#include <limits>
#include <map>
#include "rtdlite.h"

#define max(x, y) (x > y ? x : y)

namespace {
std::pair<int, int> CanonicalEdge(int u, int v)
{
    if (u > v)
        std::swap(u, v);
    return std::make_pair(u, v);
}

std::vector<double> BuildRtdlRewards(IEnv* env)
{
    std::vector<double> step_rewards(env->act_seq.size(), 0.0);
    if (!env || !env->graph)
        return step_rewards;

    const int n = env->graph->num_nodes;
    if (n <= 1 || (int)env->action_list.size() != n || env->act_seq.empty())
        return step_rewards;

    const rtd_value_t inf = std::numeric_limits<rtd_value_t>::infinity();
    std::vector<rtd_value_t> r1((size_t)n * n);
    std::vector<rtd_value_t> r2((size_t)n * n, inf);
    std::vector<int> predecessor(n, -1);
    std::map< std::pair<int, int>, Dtype > tour_edge_len;
    std::map< std::pair<int, int>, Dtype > mst_edge_len;

    for (int i = 0; i < n; ++i)
    {
        r2[(size_t)i * n + i] = 0.0;
        for (int j = 0; j < n; ++j)
            r1[(size_t)i * n + j] = env->graph->dist[i][j];
    }

    for (int i = 0; i < n; ++i)
    {
        int u = env->action_list[i];
        int v = env->action_list[(i + 1) % n];
        auto e = CanonicalEdge(u, v);
        auto w = env->graph->dist[u][v];
        predecessor[v] = u;
        tour_edge_len[e] = w;
        r2[(size_t)u * n + v] = w;
        r2[(size_t)v * n + u] = w;
    }

    auto rtdl_result = rtd_lite_run_matrix(r1.data(), r2.data(), n, true);
    for (rtd_index_t i = 0; i < rtdl_result.right_bars; ++i)
    {
        auto tour_e = CanonicalEdge((int)rtdl_result.right_to_left[i].death_i,
                                    (int)rtdl_result.right_to_left[i].death_j);
        if (!tour_edge_len.count(tour_e))
            continue;

        auto mst_e = CanonicalEdge((int)rtdl_result.right_to_left[i].birth_i,
                                   (int)rtdl_result.right_to_left[i].birth_j);
        auto mst_len = env->graph->dist[mst_e.first][mst_e.second];
        auto it = mst_edge_len.find(tour_e);
        if (it == mst_edge_len.end() || mst_len < it->second)
            mst_edge_len[tour_e] = mst_len;
    }
    rtd_lite_result_free(&rtdl_result);

    for (size_t t = 0; t < env->act_seq.size(); ++t)
    {
        int node = env->act_seq[t];
        if (node < 0 || node >= n)
            continue;
        int prev = predecessor[node];
        if (prev < 0)
            continue;

        auto e = CanonicalEdge(prev, node);
        if (!tour_edge_len.count(e) || !mst_edge_len.count(e))
            continue;
        auto complexity = tour_edge_len[e] - mst_edge_len[e];
        if (complexity < 0.0)
            complexity = 0.0;
        step_rewards[t] = -cfg::rtdl_reward_scale * complexity / env->norm;
    }
    return step_rewards;
}
}

std::vector< std::shared_ptr<Graph> > NStepReplayMem::graphs;
std::vector<int> NStepReplayMem::actions;
std::vector<double> NStepReplayMem::rewards;
std::vector< std::vector<int> > NStepReplayMem::states;
std::vector< std::vector<int> > NStepReplayMem::s_primes;
std::vector<bool> NStepReplayMem::terminals;
int NStepReplayMem::current;
int NStepReplayMem::count;
int NStepReplayMem::memory_size;
std::default_random_engine NStepReplayMem::generator;
std::uniform_int_distribution<int>* NStepReplayMem::distribution;

void NStepReplayMem::Init(int _memory_size)
{
    memory_size = _memory_size;
    graphs.resize(memory_size);
    actions.resize(memory_size);
    rewards.resize(memory_size);
    states.resize(memory_size);
    s_primes.resize(memory_size);
    terminals.resize(memory_size);

    current = 0;
    count = 0;
    distribution = new std::uniform_int_distribution<int>(0, memory_size - 1);
}

void NStepReplayMem::Clear()
{
    current = count = 0;
}

void NStepReplayMem::Add(std::shared_ptr<Graph> g, 
                        std::vector<int>& s_t,
                        int a_t, 
                        double r_t,
                        std::vector<int>& s_prime,
                        bool terminal)
{
    graphs[current] = g;
    actions[current] = a_t;
    rewards[current] = r_t;
    states[current] = s_t;
    s_primes[current] = s_prime;
    terminals[current] = terminal;

    count = max(count, current + 1);
    current = (current + 1) % memory_size; 
}

void NStepReplayMem::Add(IEnv* env)
{
    assert(env->isTerminal());
    int num_steps = env->state_seq.size();
    assert(num_steps);

    if (cfg::use_rtdl_reward)
    {
        auto topo_rewards = BuildRtdlRewards(env);
        assert(topo_rewards.size() == env->reward_seq.size());
        env->reward_seq = topo_rewards;
    }

    env->sum_rewards[num_steps - 1] = env->reward_seq[num_steps - 1];
    for (int i = num_steps - 1; i >= 0; --i)
        if (i < num_steps - 1)
            env->sum_rewards[i] = env->sum_rewards[i + 1] + env->reward_seq[i];

    for (int i = 0; i < num_steps; ++i)
    {
        bool term_t = false;
        double cur_r;
        std::vector<int>* s_prime; 
        if (i + cfg::n_step >= num_steps)
        {
            cur_r = env->sum_rewards[i];
            s_prime = &(env->action_list);
            term_t = true;
        } else {
            cur_r = env->sum_rewards[i] - env->sum_rewards[i + cfg::n_step];
            s_prime = &(env->state_seq[i + cfg::n_step]);
        }
        Add(env->graph, env->state_seq[i], env->act_seq[i], cur_r, *s_prime, term_t);
    }
}

void NStepReplayMem::Sampling(int batch_size, ReplaySample& result)
{
    assert(count >= batch_size);

    result.g_list.resize(batch_size);
    result.list_st.resize(batch_size);
    result.list_at.resize(batch_size);
    result.list_rt.resize(batch_size);
    result.list_s_primes.resize(batch_size);
    result.list_term.resize(batch_size);
    auto& dist = *distribution;
    for (int i = 0; i < batch_size; ++i)
    {
        int idx = dist(generator) % count;
        result.g_list[i] = graphs[idx];
        result.list_st[i] = &(states[idx]);
        result.list_at[i] = actions[idx];
        result.list_rt[i] = rewards[idx];
        result.list_s_primes[i] = &(s_primes[idx]);
        result.list_term[i] = terminals[idx];
    }
}
