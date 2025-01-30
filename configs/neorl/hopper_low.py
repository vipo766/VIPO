from copy import deepcopy
from configs.neorl.default import default_args


hopper_v3_low_args = deepcopy(default_args)
hopper_v3_low_args["rollout_length"] = 5
hopper_v3_low_args["penalty_coef"] = 2.5
hopper_v3_low_args["auto_alpha"] = True
hopper_v3_low_args["phi"] = 0.01 # 0.03
hopper_v3_low_args["real_ratio"] = 0.5