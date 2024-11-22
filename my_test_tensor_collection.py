'''
Example of how to create a PandasTensorCollection
'''

tensor_coll = False

if tensor_coll:
    import pandas as pd
    import torch

    from src.megapose.utils.tensor_collection import TensorCollection, PandasTensorCollection


    tc1 = TensorCollection(tensor1=torch.randn(5, 3), tensor2=torch.randn(5, 2))
    tc2 = TensorCollection(tensor1=torch.randn(3, 3), tensor2=torch.randn(2, 2))

    infos1 = pd.DataFrame({'yomama': [1, 2, 3, 4, 'nudl']})
    infos2 = pd.DataFrame({'id': [6, 7, 8]})

    ptc1 = PandasTensorCollection(infos1, tensor1=tc1.tensor1, tensor2=tc1.tensor2)
    ptc2 = PandasTensorCollection(infos2, tensor1=tc2.tensor1, tensor2=tc2.tensor2)

    print(ptc1)

else:
    