import torch
import inspect
import logging
import importlib

logger = logging.getLogger(__name__)



def load_model(args):
    # retrieve model skeleton
    model_class = importlib.import_module('..models', package=__package__).__dict__[args.model_name]

    # get required model arguments
    required_args = inspect.getfullargspec(model_class)[0]

    # collect eneterd model arguments
    model_args = {}
    for argument in required_args:
        if argument == 'self': 
            continue
        model_args[argument] = getattr(args, argument)

    # get model instance
    model = model_class(**model_args)

    # adjust arguments if needed
    if args.use_pt_model:
        if hasattr(model, 'num_embeddings'):
            args.num_embeddings = model.num_embeddings
        if hasattr(model, 'embedding_size'):
            args.embedding_size = model.embedding_size
        if hasattr(model, 'hidden_size'):
            args.hidden_size = model.hidden_size
        if hasattr(model, 'dropout'):
            args.dropout = model.dropout
    return model, args
