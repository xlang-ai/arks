class DataTypes:
    import tensorflow as tf 
    support_type = [ 'bfloat16', 'bool', 'complex', 'complex128', 'complex64', 'double', 'float16', 'float32', 'float64', 'half', 'int16', 'int32', 'int64', 'int8', 'qint16', 'qint32', 'qint8', 'quint16', 'quint8', 'string', 'uint16', 'uint32', 'uint64', 'uint8']
    type_id = list(range(len(support_type)))
    project_type = [tf.bfloat16, tf.bool, tf.complex64, tf.complex128, tf.complex64, tf.float64, tf.float16, tf.float32, tf.float64, tf.float16, tf.int16, tf.int32, tf.int64, tf.int8, tf.qint16, tf.qint32, tf.qint8, tf.quint16, tf.quint8,  tf.string, tf.uint16, tf.uint32, tf.uint64, tf.uint8]
    def __init__(self, d_types) -> None:
        d_types = str(d_types)

        if isinstance(d_types, str):
            self.true_type = self.project_type[self.support_type.index(d_types)]

    def __eq__(self, other):

        return self.true_type == other.true_type

    def __repr__(self) -> str:
        return self.true_type.__repr__()

bfloat16 = DataTypes('bfloat16')

bool = DataTypes('bool')

complex = DataTypes('complex')

complex128 = DataTypes('complex128')

complex64 = DataTypes('complex64')

double = DataTypes('double')

float16 = DataTypes('float16')

float32 = DataTypes('float32')

float64 = DataTypes('float64')

half = DataTypes('half')

int16 = DataTypes('int16')

int32 = DataTypes('int32')

int64 = DataTypes('int64')

int8 = DataTypes('int8')

qint16 = DataTypes('qint16')

qint32 = DataTypes('qint32')

qint8 = DataTypes('qint8')

quint16 = DataTypes('quint16')

quint8 = DataTypes('quint8')

string = DataTypes('string')

uint16 = DataTypes('uint16')

uint32 = DataTypes('uint32')

uint64 = DataTypes('uint64')

uint8 = DataTypes('uint8')