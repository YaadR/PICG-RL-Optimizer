#%%
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy import optimize
from tqdm import tqdm
import numpy as np
# create a numpy.interpolate example
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import random
import os
# import optax
#%%
class Environment:
    def __init__(self, bls_atempts=10):
        self.w = random.uniform(1e-03, 1e-02)
        self.delta = random.uniform(1e-03, 1e-02)
        self.alpha = 1
        self.bls_atempts=bls_atempts
        self.G = grad(self.O)
        self.delta_array = []
        self.w_array = []
        self.terminal_counter = 0
        self.terminal_consecutive_index = 0
        self.reset()
        
    def reset(self):
        mp = jnp.linspace(0, 10, num=11, endpoint=True)
        xp = jnp.cos(-mp**2/9.0)
        f = interp1d(mp, xp, kind='cubic')
        m = jnp.linspace(0, 10, num=50, endpoint=True)

        self.zero_state = f(m)
        gap = (max(self.zero_state)-min(self.zero_state))
        self.valid_range = [gap*1.1, gap*0.66]
        self.objective_prev = 0
        self.state=f(m)
        self.w = random.uniform(1e-03, 1e-02)
        self.delta = random.uniform(1e-03, 1e-02)
        self.delta_array = [self.delta]
        self.w_array = [self.w]
        self.terminal_counter = 0
        self.terminal_consecutive_index = 0

    def scaler(self,e):
        return (e * 50).round().astype(int)

    # Creating data
    def create_data(self):
        mp = jnp.linspace(0, 10, num=11, endpoint=True)
        xp = jnp.cos(-mp**2/9.0)
        f = interp1d(mp, xp, kind='cubic')
        m = jnp.linspace(0, 10, num=50, endpoint=True)
        x0=f(m)

        # Plotting the original data and the interpolations
        return mp,xp,f,m,x0

    def sig(self,x,t):
        return x**2 / (x**2 + t)

    def O(self,x, x0, t, w):
        return jnp.sum(self.sig(jnp.diff(x), t)) + w * jnp.sum((x - x0)**2)

    def backtracking_line_search(self,f, x, d, alpha, max_iter):
        for i in range(max_iter):
            xp = x + alpha * d
            if f(xp) < f(x):
                return xp
            alpha = 0.5 * alpha
        return xp

    # Function to update the plot
    def update_plot(self,xi, line):
        line.set_ydata(xi)
        plt.draw()
        plt.pause(0.01)

    def step(self, action):
        self.objective_prev = self.O(self.state,self.zero_state,self.delta,self.w)
        self.delta*=action[0]
        self.w*= action[1] 
        self.delta_array.append(self.delta) 
        self.w_array.append(self.w)
        d = -self.G(self.state, self.zero_state, self.delta, self.w)
        xi = self.backtracking_line_search(lambda x: self.O(x, self.zero_state, self.delta, self.w), self.state, d, self.alpha, self.bls_atempts)
        self.state = xi
        next_state = np.array(xi)

        return next_state

    def terminal_state(self,index,reward,num_rounds,state,state_new,objective):
        if self.terminal_consecutive_index==0:
            self.terminal_consecutive_index = index-1

        if index == num_rounds-1:
            done=1
            reward = 50
            return reward,done
        elif (np.mean((state-state_new) ** 2)<0.4) and (objective < 4.5) and (index == self.terminal_consecutive_index +1):
            self.terminal_consecutive_index = index
            self.terminal_counter+=1
            done=0
            if self.terminal_counter>=50:
                print("Success!")
                done=1
                reward = 50
                return reward,done
        else:
            done = 0
            self.terminal_consecutive_index = 0
            self.terminal_counter=0
        return reward,done

  #%%

if __name__ == "__main__":
    env = Environment()
    mp,xp,f,m,x0 = env.create_data()
    plt.plot(mp, xp, 'o', m, x0, '-')
    plt.show()
    # plot the sigmoid function for multiple values of t
    x = np.linspace(0, 10, 100)
    t = np.linspace(0.00001, 1, 10)
    y = vmap(env.sig, in_axes=(None, 0))(x, t)
    plt.plot(x, y.T)
    plt.show()
    

  
    # test
    # energy(np.diff(yt),1e-10) # should be close to 99 because np.diff(yt) has 99 elements and they are all different so sig(x,1e-10) is 1 for all of them
    G = grad(env.O)

    # make a vector is repeated values using kron
    x1 = jnp.kron(jnp.array([1, 2, 3]),jnp.ones(3)) # x1 = [1., 1., 1., 2., 2., 2., 3., 3., 3.]
    plt.plot(x1)
    plt.show()
    print(env.O(x1,x1, 1e-10,1))  # should be close to 2 because there are two jumps
    print(G(x1, x1, 1e-10,1))  # should be close to 0 because vector is already pieacewise constant
    print(env.O(x1,x1, 1e6,1e6)) # should be close to 0 because delta and the proximity weight are very large


    #%%
    xi = x0
    G=grad(env.O)
    delta = 1e-2
    w = 0.01
    # Set up the plot
    fig, ax = plt.subplots()
    line1, = ax.plot(m,x0, '-', label='Original')
    line2, = ax.plot(m, xi, '-', label='Optimized')
    ax.plot(mp, xp, 'o', label='Data Points')
    ax.legend()

    # for i in tqdm(range(1000), desc="Progress"):
    for i in range(1000):
        d = -G(xi, x0, delta, w)
        print(env.O(xi, x0, delta, w),end='\r')
        alpha = 1
        xi = env.backtracking_line_search(lambda x: env.O(x, x0, delta, w), xi, d, alpha, 10)
        env.update_plot(xi, line1)
    # plt.plot(xi)
    plt.plot(mp, xp, 'o', m, x0, '-', m, xi, '-')

    # Save the plot to the output folder
    plot_filename = os.path.join('output', 'final_plot.png')
    plt.savefig(plot_filename)

    # Optionally, show the plot
    plt.show()
# %%
