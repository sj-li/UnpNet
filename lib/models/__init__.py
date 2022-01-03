from .UnpNet import UnpNet

def get_model(model):
    return eval(model)
