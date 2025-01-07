import numpy as np

def tensor_product(*tensors):
    """
    Compute the tensor product of multiple tensors.
    
    Parameters:
        *tensors (list of np.ndarray): List of tensors to compute the tensor product.
    
    Returns:
        np.ndarray: Resulting tensor after tensor product.
    """
    result = tensors[0]
    for tensor in tensors[1:]:
        result = np.kron(result, tensor)
    return result

def gate_multiplication(*tensors):
    """
    Compute the matrix product of multiple tensors in order.
    
    Parameters:
        *tensors (list of np.ndarray): List of tensors to compute the tensor product.
    
    Returns:
        np.ndarray: Resulting tensor after matrix multiplication.
    """
    result = tensors[0]
    for tensor in tensors[1:]:
        result = np.dot(result, tensor)
    return result