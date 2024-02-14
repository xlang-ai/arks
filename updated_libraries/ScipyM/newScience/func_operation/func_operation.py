from typing import Any, Union, List
import numpy as np


def linear_interpolate(
    x,
    y,
    new_x,
    default_value=None,
    axis=None
):
    from scipy import interpolate
    x = np.asarray(x)
    y = np.asarray(y)
    new_x = np.asarray(new_x)

    y_shape = y.shape 
    if len(y_shape) > 1 and y_shape[1] > 1:
        result = interpolate.griddata(x, y, new_x, fill_value=default_value if default_value is not None else np.nan)
    else:
        result = interpolate.interp1d(x, y, axis=axis, fill_value=default_value if default_value is not None else np.nan)(new_x)
    
    return result 



def nonlinear_fit(func,
    x,
    y,
    initial_guess=None,
    check_valid=True,
    min_val=None,
    max_val=None,
    method="lm"):
    import scipy.optimize as opt 
    result = opt.curve_fit(func, x, y, initial_guess, check_finite=check_valid, bounds=(min_val, max_val), method=method)
    return result

def definite_integrate(func, start, end, args=()):
    import scipy.integrate as inte 
    result = inte.quad(func, start, end, args)
    return result[0]

def optimize(func: Any, initial_guess: Any, mode='max', args: Any = (), min_var=None, max_var=None, method: Any = None):
    import scipy.optimize as opt 
    if min_var is None or np.isscalar(min_var):
        lower_bounds = [min_var for i in range(len(initial_guess))]
    else:
        lower_bounds = min_var
    if max_var is None or np.isscalar(max_var):
        upper_bounds = [max_var for i in range(len(initial_guess))]
    else: 
        upper_bounds = max_var
    if mode == 'max':
        def cur_func(x):
            result = func(x)
            return -result 
        solution = opt.minimize(cur_func, initial_guess, args, bounds=[(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))], method=method)
    elif mode == 'min':
        solution = opt.minimize(func, initial_guess, args, bounds=[(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))], method=method)
    else:
        raise ValueError('Mode only support min or max!')
    return solution 

def scalar_minimum(func, bounds=None, extra_args=()):
    import scipy.optimize as opt 
    if bounds is None:
        bounds = [-0x7fffffff, 0x7fffffff]
    if len(bounds) != 2 or (hasattr(bounds[0], "__len__") or hasattr(bounds[1], "__len__")):
        raise ValueError('Optimization bounds must be array scalars of ONLY 2 elements each.')
    result = opt.fminbound(func, bounds[0], bounds[1], extra_args)
    return result

def ode_solver(func, x_span, y0, x_eval=None, args=(), method='RK45'):
    import scipy.integrate as inte 
    result = inte.solve_ivp(func, x_span, y0, method, x_eval, args=args)
    return result

def numeric_integrate(y, x, h=0.1, axis=-1) -> Union[np.ndarray, float]:
    import scipy.integrate as inte
    result = inte.trapz(y, x, h, axis)
    return result


class NDLinearInterpolator:
    def __init__(self, variable, covariable, default_value=None, axis=None) -> None:
        self.variable = np.asarray(variable)
        self.covariable = np.asarray(covariable)
        if default_value is None:
            self.default_value = np.nan 
        else:
            self.default_value = default_value
        from scipy import interpolate
        y_shape = self.covariable.shape 
        if len(y_shape) == 1:
            if len(self.variable.shape) == 2:
                self.interpolator = lambda new_x: interpolate.griddata(self.variable, self.covariable, new_x, fill_value=self.default_value)
            else: 
                self.interpolator = interpolate.interp1d(self.variable, self.covariable, axis=axis, fill_value=self.default_value)
        else:
            self.interpolator = interpolate.interp1d(self.variable, self.covariable, axis=axis, fill_value=self.default_value)
        
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.interpolator(* args, ** kwds)
    

class NDSplineInterpolator: 
    def __init__(self, *variable,
        covariable,
        weights = None,
        ks: List[int] = [3],
        s: int = 0,) -> None:
        self.variable_list = list(variable)
        self.covariable = covariable
        self.weights = weights 
        self.ks = ks 
        self.s = s 
        
        from scipy import interpolate
        if len(self.variable_list) == 1:
            self.interpolator = interpolate.UnivariateSpline(self.variable_list[0], covariable,
                                                             weights, k=ks[0], s=self.s)
        elif len(self.variable_list) == 2:
            if (len(ks) == 1):
                self.ks = [ks[0]] * 2

            self.interpolator = interpolate.RectBivariateSpline(self.variable_list[0], 
                                                                self.variable_list[1],
                                                                covariable, kx=self.ks[0], ky=self.ks[1],s=self.s)
        else: 
            raise ValueError('Not supported number of variables')
    
    def __call__(self, *new_x, **kwargs):
        result = self.interpolator(*new_x, **kwargs)
        return result