from .nq_transf_learner import NQTransfLearner
from .cq_learner import CQLearner

REGISTRY = {}

REGISTRY["nq_transf_learner"] = NQTransfLearner
REGISTRY["cq_learner"] = CQLearner