from copy import deepcopy
from configs.gym.default import default_args


walker2d_medium_expert_args = deepcopy(default_args)
walker2d_medium_expert_args["rollout_length"] = 1
walker2d_medium_expert_args["penalty_coef"] = 2.0
walker2d_medium_expert_args["phi"] = 0.35
walker2d_medium_expert_args["real_ratio"] = 0.5