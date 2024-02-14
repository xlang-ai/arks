import numpy as np 
from typing import Any, Dict, List, Optional, Union


class SparseMatrix:
    def __init__(self, input, form='coo') -> None:
        import scipy.sparse as sp
        self.form = form 

        if form == 'coo':
            self.sparse_matrix = sp.coo_matrix(input)
        if form == 'csr':
            self.sparse_matrix = sp.csr_matrix(input)

        if form == 'lil':
            self.sparse_matrix = sp.lil_matrix(input)
        if form == 'csc':
            self.sparse_matrix = sp.csc_matrix(input)
        if form == 'dia':
            self.sparse_matrix = sp.dia_matrix(input)
        if form == 'dok':
            self.sparse_matrix = sp.dok_matrix(input)
            
    @property 
    def keys(self):
        return self.sparse_matrix.keys() if self.form == 'dok' else None 
    
    @property 
    def values(self):
        return np.array(list(self.sparse_matrix.values())) if self.form == 'dok' else None 

    @property
    def data(self):
        return self.sparse_matrix.data
    
    @property
    def shape(self):
        return self.sparse_matrix.shape 
    
    @property 
    def T(self):
        return SparseMatrix(self.sparse_matrix.T, form=self.form)
    
    @property 
    def ind_ptr(self):
        return self.sparse_matrix.indptr if self.form == 'csr' or self.form == 'csc' else None 
    
    @property
    def num_ptr(self):
        return self.sparse_matrix.num_ptr if self.form == 'csr' else None
    
    def __ne__(self, __o: object):
        if isinstance(__o, SparseMatrix):
            return SparseMatrix(self.sparse_matrix != __o.sparse_matrix, form=self.form)
        else:
            return False
        
    
    def __eq__(self, __o: object):
        if isinstance(__o, SparseMatrix):
            return SparseMatrix(self.sparse_matrix == __o.sparse_matrix, form=self.form)
        else:
            return False
    
    def __getitem__(self, idx):
        return self.sparse_matrix[idx]
    
    def __setitem__(self, idx, value):
        self.sparse_matrix[idx] = value 
    
    def update(self, keys, values):
        if self.form != "dok":
            return 
        if not isinstance(keys, type({}.keys())):
            raise TypeError("Not supported keys type!")
        values = np.asarray(values)
        self.sparse_matrix._update(zip(keys, values))
    
    def diag(self, k: int=0):
        return self.sparse_matrix.diagonal(k)
    
    def matmul(self, other):
        if isinstance(other, np.ndarray):
            return self.sparse_matrix.dot(other)
        
        return self.sparse_matrix.dot(other.sparse_matrix)
    
    def max(self, axis=None):
        return self.sparse_matrix.max(axis)
    
    def min(self, axis=None):
        return self.sparse_matrix.min(axis)
    
    def mean(self, axis=None):
        return self.sparse_matrix.mean(axis)
    
    def __repr__(self):
        return self.sparse_matrix.__repr__()
    
    def __str__(self):
        return self.sparse_matrix.__str__()
    
    def std(self, axis=None):
        mean = self.mean(axis=axis)
        N = self.shape[0]
        sqr = self.sparse_matrix.copy()
        sqr.data **= 2
        std = np.sqrt(sqr.sum(axis) / N - mean ** 2)
        return std
    
    def sum(self, axis=None):
        return self.sparse_matrix.sum(axis)
    
    @classmethod 
    def cvt_scipy(cls, scipy_sparse, form):
        result = SparseMatrix(scipy_sparse.toarray(), form) 
        return result
    
    @classmethod 
    def hori_cat(cls, A, B, dtype=None):
        import scipy.sparse as sp
        assert isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix) and B.form == A.form
        form = A.form 
        result = sp.hstack([A.sparse_matrix, B.sparse_matrix], dtype=dtype)
        return SparseMatrix.cvt_scipy(result, form)
        
    def hcat(self, other, dtype=None):
        import scipy.sparse as sp
        assert isinstance(other, SparseMatrix) and self.form == other.form
        form = self.form 
        result = sp.hstack([self.sparse_matrix, other.sparse_matrix], dtype=dtype)
        return self.cvt_scipy(result, form)
    
    @classmethod 
    def vert_cat(cls, A, B, dtype=None):
        import scipy.sparse as sp
        assert isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix) and B.form == A.form
        form = A.form 
        result = sp.vstack([A.sparse_matrix, B.sparse_matrix], dtype=dtype)
        return SparseMatrix.cvt_scipy(result, form)
        
    def vcat(self, other, dtype=None):
        import scipy.sparse as sp
        assert isinstance(other, SparseMatrix) and self.form == other.form
        form = self.form 
        result = sp.vstack([self.sparse_matrix, other.sparse_matrix], dtype=dtype)
        return self.cvt_scipy(result, form)
        
    def sparsify(self):
        self.sparse_matrix.eliminate_zeros()
    
    def filled_ind(self):
        
        return self.sparse_matrix.nonzero()

    def empty_num(self):
        return self.sparse_matrix.count_nonzero()
    
    def filled_num(self):
        result = self.filled_ind()
        return len(result[0])
    
    def to_numpy(self):
        return self.sparse_matrix.toarray()
    
    def to_csr_form(self):
        if self.form == 'csr':
            return
        self.sparse_matrix = self.sparse_matrix.tocsr()
        
        self.form = 'csr'
    
    def to_dok_form(self):
        if self.form == 'dok':
            return 
        
        self.sparse_matrix = self.sparse_matrix.todok()
        self.form = "dok"
    
    def to_csc_form(self):
        if self.form == 'csc':
            return
        self.sparse_matrix = self.sparse_matrix.tocsc()
        self.form == 'csc'
    
    def to_coo_form(self):
        if self.form == 'coo':
            return
        self.sparse_matrix = self.sparse_matrix.tocoo()
        self.form == 'coo'
    
    def to_lil_form(self):
        if self.form == 'lil':
            return
        self.sparse_matrix = self.sparse_matrix.tolil()
        self.form == 'lil'
    
    def to_dia_form(self):
        if self.form == 'dia':
            return 
        self.sparse_matrix = self.sparse_matrix.todia()
        self.form = 'dia'
    
    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return SparseMatrix(self.sparse_matrix + other , form=self.form)
        if isinstance(other, SparseMatrix):
            return SparseMatrix(self.sparse_matrix + other.sparse_matrix, form=self.form)
        if np.isscalar(other):
            return SparseMatrix(self.sparse_matrix.data + other, form=self.form)
        raise TypeError('Not supported addition type!')
    
    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return SparseMatrix(self.sparse_matrix * other, form=self.form)
        if isinstance(other, SparseMatrix):
            return SparseMatrix(self.sparse_matrix.multiply(other.sparse_matrix), form=self.form)
        raise TypeError('Not supported multiplication type!')
    
    def getrow(self, rows):
        result = self.sparse_matrix.getrow(rows)
        return SparseMatrix(result, form=self.form)

    def set_diag(self, val, k=0):
        self.sparse_matrix.setdiag(val, k=k)



def random_fill(row, col, filled_percent=0.1, form='csr', dtype=None, random_seed=-1):
    import scipy.sparse as sp 
    result = sp.random(row, col, filled_percent, form, random_state=random_seed, dtype=dtype)
    return SparseMatrix(result, form=form)       

def diag(mat: SparseMatrix,
    k: int = 0):
    return mat.diag(k=k)


def sparse_from_diag(data: Any, diags: Any, row: Any, col: Any) -> SparseMatrix:
    import scipy.sparse as sp 
    result = sp.spdiags(data, diags, row, col)
    cur_result = SparseMatrix.cvt_scipy(result, form='dia')
    return cur_result

def sparse_multiply(a: SparseMatrix, b: SparseMatrix):
    return a * b