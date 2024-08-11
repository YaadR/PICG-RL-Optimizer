#%%
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy import optimize
from tqdm import tqdm
import numpy as np

# import optax
#%%
# create a numpy.interpolate example
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Creating data
mp = jnp.linspace(0, 10, num=11, endpoint=True)
xp = jnp.cos(-mp**2/9.0)

f = interp1d(mp, xp, kind='cubic')

m = jnp.linspace(0, 10, num=100, endpoint=True)
x0=f(m)

# Plotting the original data and the interpolations
plt.plot(mp, xp, 'o', m, x0, '-')
plt.show()

#%%

def sig(x,t):
    return x**2 / (x**2 + t)

# plot the sigmoid function for multiple values of t
x = np.linspace(0, 10, 100)
t = np.linspace(0.00001, 1, 10)
y = vmap(sig, in_axes=(None, 0))(x, t)
plt.plot(x, y.T)
plt.show()

def O(x, x0, t, w):
    return jnp.sum(sig(jnp.diff(x), t)) + w * jnp.sum((x - x0)**2)
#%%
# test
# energy(np.diff(yt),1e-10) # should be close to 99 because np.diff(yt) has 99 elements and they are all different so sig(x,1e-10) is 1 for all of them
G = grad(O)
# make a vector is repeated values using kron
x0 = jnp.kron(jnp.array([1, 2, 3]),jnp.ones(3)) # x0 = [1., 1., 1., 2., 2., 2., 3., 3., 3.]
plt.plot(x0)
plt.show()
print(O(x0,x0, 1e-10,1))  # should be close to 2 because there are two jumps
print(G(x0, x0, 1e-10,1))  # should be close to 0 because vector is already pieacewise constant
print(O(x0,x0, 1e6,1e6)) # should be close to 0 because delta and the proximity weight are very large


#%%

def backtracking_line_search(f, x, d, alpha, max_iter):
    for i in range(max_iter):
        xp = x + alpha * d
        if f(xp) < f(x):
            return xp
        alpha = 0.5 * alpha
    return xp

xi = x0

G=grad(O)
delta = 1e-2
w = 0.01

for i in tqdm(range(1000), desc="Progress"):
    d = -G(xi, x0, delta, w)
    alpha = 1
    xi = backtracking_line_search(lambda x: O(x, x0, delta, w), xi, d, alpha, 10)
    
# plt.plot(xi)
plt.plot(mp, xp, 'o', m, x0, '-', m, xi, '-')
# %%
