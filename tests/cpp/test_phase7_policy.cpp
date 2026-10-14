#include "slbjnv/branch_and_price/slbjnv_branching_policy.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <vector>

namespace {

class IncreasingPredictor final : public slbjnv::ScorePredictor {
public:
    void load_model(const std::filesystem::path&) override {}

    std::vector<double> predict_scores(
        const std::vector<slbjnv::FeatureExtractor::FeatureVector>& features) override {
        count = features.size();
        std::vector<double> result(count);
        for (std::size_t i = 0; i < count; ++i) result[i] = static_cast<double>(i);
        return result;
    }

    std::size_t count{};
};

slbjnv::GBDInstance instance() {
    slbjnv::GBDInstance value;
    value.instance_id = "joint_policy";
    value.items = {{0, 1.0, 4.0}, {1, 1.0, 5.0}, {2, 1.0, 6.0}};
    value.machines = {{0, 3.0}, {1, 3.0}};
    value.compatible_with_master.assign(3, std::vector<bool>(3, true));
    value.validate_or_throw();
    return value;
}

slbjnv::BranchNode node(slbjnv::NodeId id, std::size_t exploration_index) {
    slbjnv::BranchNode value;
    value.id = id;
    value.parent_id = id == 0 ? slbjnv::kNoNode : 0;
    value.depth = static_cast<std::size_t>(id);
    value.columns = {{0, 0, 0, {0, 1}, 9.0}, {1, 1, 1, {1, 2}, 11.0}};
    value.relaxation.status = slbjnv::SolveStatus::optimal;
    value.relaxation.converged = true;
    value.relaxation.master.lp.status = slbjnv::SolveStatus::optimal;
    value.relaxation.master.lp.objective = 10.0;
    value.relaxation.master.lp.primal = {0.5, 0.5};
    value.relaxation.first_iteration_duals.item_duals.assign(3, 0.0);
    value.relaxation.first_iteration_duals.machine_duals.assign(2, 0.0);
    value.upper_bound = 10.0;
    value.exploration_index = exploration_index;
    value.active = true;
    return value;
}

}  // namespace

int main() {
    auto problem = instance();
    slbjnv::SearchTree tree;
    tree.add_node(node(0, 0));
    tree.add_node(node(1, 1));
    IncreasingPredictor predictor;
    slbjnv::SLbJNVBranchingPolicy policy(problem, tree, predictor);
    const auto decision = policy.select({&tree.node(0), &tree.node(1)});
    assert(decision.has_value());
    assert(predictor.count > 2);
    assert(decision->node_id == 1);
    assert(std::abs(decision->score - static_cast<double>(predictor.count - 1)) < 1.0e-12);
    assert(policy.predicted_pairs() == predictor.count);
    assert(policy.inference_seconds() >= 0.0);
    return 0;
}
