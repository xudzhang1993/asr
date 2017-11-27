class Tag(object):
    train_loss = "train_loss"
    dev_loss = "dev_loss"
    test_loss = "test_loss"


class LogWriter(object):
    def __init__(self, log_file):
        self.log_file = log_file

    def __enter__(self):
        self.handle = open(self.log_file, 'a')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handle.close()

    def write(self, value, tag):
        self.handle.write(tag + ":" + value + "\n")
