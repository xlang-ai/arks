from typing import Any
import numpy as np 
import jumptensor.src.dataTypes as dT
class TensorVariable:
    def __init__(self, var, *args, **kwargs):
        self.var = var 
        import tensorflow as tf 
        self.cur = tf.Variable(self.var)
        self.shape = self.cur.shape
        self.dtype = dT.DataTypes(self.cur.numpy().dtype)

    def assn_value(self,  value,
                    is_locked=False,
                    name=None,
                    return_value=False):
        self.var = value
        ret = self.cur.assign(value, is_locked, name, return_value)
        if return_value:
            return TensorVariable(ret.numpy())
        else:
            return None
    
    def numpy_value(self):
        import tensorflow as tf 
        cur = tf.Variable(self.var)
        return cur.numpy()
    
    def __repr__(self):
        
        return self.var.__repr__()
    
    def __eq__(self, other):
        if isinstance(other, TensorVariable):
            return Tensor(self.var == other.var)
        else:
            return Tensor(self.var == other)
        

class Tensor:
    def __init__(self, t, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import tensorflow as tf
        self.t: np.ndarray 
        if type(t) == tf.Tensor:
            self.t = t.numpy()
        elif type(t) == Tensor:
            self.t = t.numpy_value()
        else:
            self.t = np.array(t, dtype=dtype)
        self.shape = self.t.shape
        self.dtype = dT.DataTypes(self.t.dtype)
    
    def rev(self, dim=None):
        import tensorflow as tf
        t = tf.constant(self.t)

        if dim is None:
            self.t = self.t[..., ::-1]
        else:
            self.t = tf.reverse(t, axis=[dim]).numpy()
    
    def numpy_value(self):
        
        return self.t
    
    def __str__(self) -> str:
        
        representation = "<jumptensor.Tensor: shape={}, dtype={},\nnumpy={}".format(self.t.shape, self.t.dtype, self.t.__repr__())
        return representation
    def __repr__(self): 
        representation = self.t.__str__()
        return representation
    
    def __getitem__(self, idx):
        result = self.t[idx]
        if hasattr(result, "__len__") and len(result) > 0:
            return Tensor(result)
        return result
        # result = Tensor(self.t[idx])
        # if len(result.shape) == 0:
        #     # if the result is a 1-d and 1-element tensor, return the value itself
        #     return result[0]
        # else:
        #     return result

    def __setitem__(self, idx, value):
        self.t[idx] = value
        
    def transform_shape(self, shape):
        return transform_shape(self.t, shape)
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.t - other.t)
        else:
            return Tensor(self.t - other)
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.t + other.t)
        else:
            return Tensor(self.t + other)
    
    def __len__(self):
        return self.t.__len__()
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.t * other.t)
        else:
            return Tensor(self.t * other)
    
    def __le__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.t <= other.t )
        else:
            return Tensor(self.t <= other)
        
        
    def __lt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.t < other.t )
        else:
            return Tensor(self.t < other)
    
    def __ge__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.t >= other.t )
        else:
            return Tensor(self.t > other)
    def __gt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.t > other.t )
        else:
            return Tensor(self.t > other)
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.t == other.t )
        else:
            return Tensor(self.t == other )
    
    def __ne__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.t != other.t)
        else:
            return Tensor(self.t != other)
        
    def __invert__(self):
        return Tensor(self.t.__invert__())
    
    def __copy__(self):
        return Tensor(self.t.copy())
    
    def get_std(self, axis, keep_dims=False):
        import tensorflow as tf 
        cur_input = tf.constant(self.t)
        return Tensor(tf.math.reduce_std(cur_input,axis=axis, keepdims=keep_dims))
    
    def get_mean(self, axis, keep_dims=False):
        import tensorflow as tf 
        cur_input = tf.constant(self.t)
        return Tensor(tf.math.reduce_mean(cur_input,axis=axis, keepdims=keep_dims))
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.t / other.t)
        else:
            return Tensor(self.t / other)
    
    def mm(self, other):
        assert isinstance(other, Tensor)
        return mm(self, other)

    

def init_variable(init_value):
    
    return TensorVariable(init_value)

def rev_tensor(input: Tensor, dim=None, inplace=None):
    import tensorflow as tf
    
    if inplace is None or inplace:
        input.rev(dim=dim)
        return None 
    else:
        cur_input = tf.constant(input.numpy_value())
        return Tensor(tf.reverse(cur_input, axis=[dim]).numpy())
    

def create_one_hot(num_classes: Any, 
                    indices: Any, 
                    hot_value: Any = None, 
                    cold_value: Any = None, 
                    axis: Any  = None, 
                    dtype: Any  = None, ):
    import tensorflow as tf
    result = tf.one_hot(indices, num_classes, hot_value, cold_value, axis=axis, dtype=dtype)

    return Tensor(result.numpy())

def init_tensor(value):

    
    return Tensor(value)

def const(value, dtype=None,
    shape=None):
    import tensorflow as tf 
    if isinstance(value, Tensor):
        return value
    cur_result = tf.constant(value, dtype=dtype.true_type if dtype is not None else None)
    return Tensor(cur_result.numpy())
    


def seq_padding_mask(
    padding_ids,
    mask_max_length: int,
    dtype = dT.bool
) -> Tensor:
    import tensorflow as tf 
    result = tf.sequence_mask(padding_ids, mask_max_length, dtype=dtype.true_type)
    return Tensor(result.numpy())

def repeated_copy(
    input,
    repeat_times
):
    import tensorflow as tf 
    cur_input = tf.constant(input.numpy_value())
    result = tf.tile(cur_input, repeat_times)
    return Tensor(result.numpy())

def pile(
    tensors,
    axis
):
    import tensorflow as tf 
    cur_input = [tf.constant(x.numpy_value()) for x in tensors]
    result = tf.stack(cur_input, axis)
    return Tensor(result.numpy())


def splice(tensors,
    axis):
    import tensorflow as tf 
    cur_input = [tf.constant(x.numpy_value()) for x in tensors]
    result = tf.concat(cur_input, axis)
    return Tensor(result.numpy())


def delete_axis(input,
    axis=None) -> Tensor:
    import tensorflow as tf
    assert isinstance(axis, int)
    cur_input = tf.constant(input.numpy_value())
    result = tf.squeeze(cur_input, axis=[axis])
    return Tensor(result.numpy())

def insert_new_axis(input,
    axis=None):
    import tensorflow as tf 
    
    cur_input = tf.constant(input.numpy_value())
    result = tf.expand_dims(cur_input, axis)
    return Tensor(result.numpy())

def transform_shape(tensor,
    shape):
    import tensorflow as tf 
    result = tf.reshape(tensor, shape)
    return Tensor(result.numpy())

def condition_filling(condition: Tensor,
    true_fill_value: Tensor,
    false_fill_value: Tensor):
    import tensorflow as tf 

    cur_condition = tf.constant(condition.numpy_value())
    cur_true = tf.constant(true_fill_value.numpy_value())
    cur_false = tf.constant(false_fill_value.numpy_value())

    result = tf.where(cur_condition, cur_true, cur_false)
    return Tensor(result.numpy())

def cull_nd(input: Any,
    indices: Any,
    batch_dims: int):
    import tensorflow as tf 
    cur_input = tf.constant(input.numpy_value())
    cur_indices=tf.constant(np.array([tf.constant(x.numpy_value()) if isinstance(x, Tensor) else x for x in indices ]))
    # print(cur_input, cur_indices, type(cur_indices))
    result = tf.gather_nd(cur_input, cur_indices, batch_dims)
    return Tensor(result.numpy())

def byte2text(
    bytes_or_text, encoding='utf-8'
    ) -> str:
    import tensorflow as tf 
    result = tf.compat.as_str(
        bytes_or_text, encoding=encoding
    )
    return result 

def text2byte(bytes_or_text, encoding='utf-8') -> bytes:
    import tensorflow as tf 
    result = tf.compat.as_bytes(
        bytes_or_text, encoding=encoding
    )
    return result


def mm(a: Tensor, b: Tensor) -> Tensor:
    import tensorflow as tf 
    cur_a = tf.constant(a.numpy_value())
    cur_b = tf.constant(b.numpy_value())
    ret = tf.matmul(cur_a, cur_b)
    return Tensor(ret.numpy())

def einsum(equation, *operands):
    import tensorflow as tf
    cur_operands = [tf.constant(x.numpy_value()) for x in operands]
    ret = tf.einsum(equation, *cur_operands)
    return Tensor(ret.numpy())

if __name__ == "__main__":


    pass 
    