from copy import deepcopy
from configs.neorl.default import default_args


hopper_v3_medium_args = deepcopy(default_args)
hopper_v3_medium_args["rollout_length"] = 5
hopper_v3_medium_args["penalty_coef"] = 1.5
hopper_v3_medium_args["auto_alpha"] = True
hopper_v3_medium_args["phi"] = 0.03 # 0.01
hopper_v3_medium_args["real_ratio"] = 0.5