
class BaseModel(object):
    def __init__(self, single_batch=False):
        self.single_batch = single_batch
        pass

class BaseEncoder(object):

    def __init__(self):
        pass

    def encode(self, inputs, batch_size):
        pass


