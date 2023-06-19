from sklearn import datasets
from skmultiflow.data import DataStream
import pandas as pd

def build_data_stream(type: str, **kwargs):
    """
    Build a stream of data.

    Parameters
    ----------
    type : str
        The type of stream to build.

    Returns
    -------
    stream : object
        The created stream
    """
    if type == 'covtype':
        dataset = datasets.fetch_covtype(data_home='./datasets', as_frame=True)
        return DataStream(data=dataset)
    else:
        raise ValueError(f'Invalid stream type: {type}')
