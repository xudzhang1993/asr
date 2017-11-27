from collections import namedtuple


class TrainOutput(namedtuple("TrainOutput", ("update", "loss"))): pass


class EvalOutput(namedtuple("EvalOutput", ("predict_id", "predict_string"))): pass


class ModelFeed(namedtuple("Input", ("feature", "target_input", "target_output", "feature_length", "target_length"))): pass


def create_none_input():
    return ModelFeed(None, None, None, None, None)
