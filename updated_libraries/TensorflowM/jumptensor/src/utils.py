
from .tensor import Tensor

from jumptensor.src import dataTypes 


class Dataset:
    def __init__(self, dataset) -> None:
        import tensorflow as tf
        self.dataset: tf.data.Dataset = dataset
    
    @classmethod 
    def create_dataset_from_tensor(cls, tensors):
        import tensorflow as tf 

        result = Dataset(tf.data.Dataset.from_tensor_slices(tensors))
        return result 
    
    def map(self, func):
        import tensorflow as tf 
        result = self.dataset.map(func)
        return Dataset(result)
    
    def map_to_flat(self, map_func):
        import tensorflow as tf 
        def cur_map_func(x):
            result = map_func(x)
            return result.dataset
        result = self.dataset.flat_map(cur_map_func)
        return Dataset(result)
    
    def as_numpy_iterator(self):
        return self.dataset.as_numpy_iterator()
    
    def __iter__(self):

        ret = [Tensor(i.numpy()) for i in iter(self.dataset)]
        return iter(ret)


class Distribution:
    def __init__(self) -> None:
        pass
    
    @classmethod 
    def uniform(cls, shape, low=0, high=None, dtype=None, seed=None, name=None):
        """Outputs random values from a uniform distribution.
        The generated values follow a uniform distribution in the range [low, high). The lower bound low is included in the range, while the upper bound high is excluded.

        For floats, the default range is [0, 1). For ints, at least high must be specified explicitly.

        In the integer case, the random integers are slightly biased unless high - low is an exact power of two. 
        The bias is small for values of high - low significantly smaller than the range of the output (either 2**32 or 2**64).


        Args:
            * shape:	A 1-D integer Tensor or Python array. The shape of the output tensor.
            * low:	A Tensor or Python value of type dtype, broadcastable with shape (for integer types, broadcasting is not supported, so it needs to be a scalar). The lower bound on the range of random values to generate (inclusive). Defaults to 0.
            * high:	A Tensor or Python value of type dtype, broadcastable with shape (for integer types, broadcasting is not supported, so it needs to be a scalar). The upper bound on the range of random values to generate (exclusive). Defaults to 1 if dtype is floating point.
            * dtype:	The type of the output: float16, float32, float64, int32, or int64.
            * seed:	A Python integer. Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.
            * name:	A name for the operation (optional).

        Returns:
            A tensor of the specified shape filled with random uniform values.
        """
        import tensorflow as tf 
        # if dtype is None:
        #     dtype = tf.dtypes.float32 
        result = tf.random.uniform(
            shape, minval=low, maxval=high, dtype=tf.dtypes.float32  if dtype is None else dtype.true_type, seed=seed, name=name
        )
        
        return Tensor(result.numpy())
    
    @classmethod 
    def normal(cls, shape, mean=0.0, stddev=1.0, dtype=None, seed=None, name=None
    ):
        """Outputs random values from a normal distribution with the given mean and standard deviation.

        Args:
            * shape:	A 1-D integer Tensor or Python array. The shape of the output tensor.
            * mean:	A Tensor or Python value of type dtype, broadcastable with stddev. The mean of the normal distribution.
            * stddev:	A Tensor or Python value of type dtype, broadcastable with mean. The standard deviation of the normal distribution.
            * dtype:	The type of the output.
            * seed:	A Python integer. Used to create a random seed for the distribution. 
            * name:	A name for the operation (optional).

        Returns:
            A tensor of the specified shape filled with random normal values.
        """
        import tensorflow as tf 

        result = tf.random.normal(shape, mean, stddev, tf.dtypes.float32  if dtype is None else dtype.true_type, seed, name)
        return Tensor(result.numpy())
    
    

def arg(
input, mode, axis=None, name=None, output_type=dataTypes.int64
):
    import tensorflow as tf
    cur_input = tf.constant(input.numpy_value())
    if mode == "max":
        result = tf.argmax(cur_input, axis, output_type.true_type, name)
    elif mode == 'min':
        result = tf.argmin(cur_input, axis, output_type.true_type, name)
    else:
        raise ValueError('Only support argmax and argmin!')
    
    return Tensor(result.numpy())


def deter_seed(seed):
    import tensorflow as tf 
    tf.random.set_seed(seed)



def get_summation(
    input: Tensor,
    axis=None,
    keep_dims=None
):
    import tensorflow as tf
    cur_input = tf.constant(input.numpy_value())
    result = tf.reduce_sum(cur_input, axis, keep_dims)
    return Tensor(result.numpy())

def get_multiplication(input: Tensor,
    axis=None,
    keep_dims=None):
    import tensorflow as tf
    cur_input = tf.constant(input.numpy_value())
    result = tf.reduce_prod(cur_input, axis, keep_dims)
    return Tensor(result.numpy())

def multiplicative_inverse(input: Tensor):
    import tensorflow as tf
    cur_input = tf.constant(input.numpy_value())
    result = tf.math.reciprocal(cur_input)
    return Tensor(result.numpy())

def get_minimum(input: Tensor, axis=None, keep_dims=None):
    import tensorflow as tf 
    cur_input = tf.constant(input.numpy_value())
    result = tf.reduce_min(cur_input, axis=axis, keepdims=keep_dims)
    return Tensor(result.numpy())

def get_reciprocal(x: Tensor):
    import tensorflow as tf
    cur_input = tf.constant(x.numpy_value()) 
    result = tf.math.reciprocal(cur_input)
    return Tensor(result.numpy())

def get_square(x: Tensor):
    import tensorflow as tf 
    cur_input = tf.constant(x.numpy_value())
    result = tf.square(cur_input)
    return Tensor(result.numpy())

def get_subtraction(x: Tensor, y: Tensor):
    import tensorflow as tf 
    cur_x = tf.constant(x.numpy_value())
    cur_y = tf.constant(y.numpy_value())
    result = tf.subtract(cur_x, cur_y)
    return Tensor(result.numpy())


def get_mean(input: Tensor, axis=None, keep_dims=False):
    return input.get_mean(axis=axis, keep_dims=keep_dims)
    

def get_std(input: Tensor, axis=None, keep_dims=False):
    return input.get_std(axis=axis, keep_dims=keep_dims)


def ones_like(ref: Tensor, dtype=None):
    import tensorflow as tf 
    cur_ref = tf.constant(ref.numpy_value())
    return Tensor(tf.ones_like(cur_ref), dtype=dtype)

def ones(shape, dtype=None):
    import tensorflow as tf 
    return Tensor(tf.ones(shape), dtype=dtype)

def zeros_like(ref: Tensor, dtype=None):
    import tensorflow as tf 
    cur_ref = tf.constant(ref.numpy_value())
    return Tensor(tf.zeros_like(cur_ref), dtype=dtype)

def zeros(shape, dtype=None):
    import tensorflow as tf 
    return Tensor(tf.zeros(shape), dtype=dtype)

def change_dtype(x, dtype, name=None):
    import tensorflow as tf 
    cur_x = tf.constant(x.numpy_value())

    ret = tf.cast(cur_x, dtype.true_type, name)

    return Tensor(ret.numpy())