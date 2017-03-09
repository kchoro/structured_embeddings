import numpy.matlib as mat
import numpy.random as random
import numpy.linalg as linalg
import numpy as np
import math

def samples(d, n): # generate a sequence of unit length vector pairs
  s = []
  for i in range(0, n):
    x1 = mat.randn((d, 1))
    x2 = mat.randn((d, 1))
    s.append((x1 / linalg.norm(x1), x2 / linalg.norm(x2)))
  return s

def expand(Q, height, k):
  for i in range(0, height):
    Q = entangle(Q, k)
  return Q

def entangle(Q, k): # Q must be an isometry
  D = mix(Q, k)
  C = mix(Q, k)
  return mat.bmat([[D, C],[C, -D]]) / math.sqrt(2.0)

# def entangle2(Q, k):
# 	xx

def mix(Q, k):
  W = mat.eye(Q.shape[1])
  for i in range(0, k):
    W *= Q * np.mat(np.diag(random_binary_array(Q.shape[1])))
  return W

# random_binary_array returns a random {-1,+1} vector  of length n
def random_binary_array(n):
  return [random.choice([-1,1]) for _ in range(n)]

# subselect projects Q onto s rows.
# normalization is applied under the assumption that Q is an isometry.
def subselect(Q, s):
  return math.sqrt(Q.shape[1] / s) * mat.mat(random.permutation(Q))[0:s]

def hadamard_kernel(deg, k):
  n = 2**deg
  H = np.mat(hadamard(n))/math.sqrt(n)
  for i in range(0, k):
    H *= H * np.mat(np.diag(random_binary_array(n)))
  return H

def recursive_kernel(deg, k):
  return mix(expand(np.mat([1]), deg, k), k)

def gaussian_kernel(deg):
  n = 2**deg
  return mat.randn(n, n) / math.sqrt(n) # gaussian?

def gaussian_orthogonal_kernel(deg):
  n = 2**deg
  rows = []
  for i in range(0, n):
    x = mat.randn(n, 1)
    x = x / linalg.norm(x)
    p = mat.zeros((1, n))
    for r in rows:
      p += (r * x) * r
    y = np.transpose(x) - p
    y = y / linalg.norm(y)
    rows.append(y)
  return np.concatenate(rows)

def kernel_sample_mse(K, x):
  return (linalg.norm(K*x)-linalg.norm(x))**2

def kernel_dot(K, x, y): # x and y are column vectors
  return np.transpose(K*x)*(K*y)

def kernel_dot_error(K, x, y):
  return kernel_dot(K, x, y) - np.transpose(x)*y

# ADDING STRUCTURE TO THE DIAGONAL CHOICES

def recursive_kernel_with_knives(deg):
  return mix_with_knives(expand_with_knives(np.mat([1]), deg))

def expand_with_knives(Q, height):
  for i in range(0, height):
    Q = entangle_with_knives(Q)
  return Q

def entangle_with_knives(Q): # Q must be an isometry
  D = mix_with_knives(Q)
  C = mix_with_knives(Q)
  return mat.bmat([[D, C],[C, -D]]) / math.sqrt(2.0)

def mix_with_knives(Q):
  n = Q.shape[1]
  W = mat.eye(n)
  pk = permuted_knives(int(math.log(n, 2)))
  for knife in pk:
    h = np.asarray(knife).flatten()
    W *= Q * np.mat(np.diag(h))
  return W

def permuted_knives(d):
  p = random.permutation(2**d)
  r = []
  for knife in knives(d):
    r.append(permute_knife(knife, p))
  return r

def permute_knife(knife, pp):
  y = []
  for p in pp:
    y.append(knife[0, p])
  return mat.mat(np.array(y))

def knives(dim): # returns a list of structured random row vectors
  if dim == 0:
    r = random.choice([-1, +1])
    return [mat.mat(r), mat.mat(-r)]
  n = int(2**dim)
  ss = [
      np.concatenate([mat.ones((1, n/2)), -mat.ones((1, n/2))], axis=1),
      np.concatenate([-mat.ones((1, n/2)), mat.ones((1, n/2))], axis=1),
  ]
  for h in knives(dim-1):
    ss.append(np.concatenate([h, h], axis=1))
  return ss

# SUBSTITUTE DIAGONAL SIGNS WITH A PERMUTATION

def perm_kernel(deg, k):
  return mix_perm(expand_perm(np.mat([1]), deg, k), k)

def expand_perm(Q, height, k):
  for i in range(0, height):
    Q = entangle_perm(Q, k)
  return Q

def entangle_perm(Q, k): # Q must be an isometry
  D = mix_perm(Q, k)
  C = mix_perm(Q, k)
  return mat.bmat([[D, C],[C, -D]]) / math.sqrt(2.0)

def mix_perm(Q, k):
  n = Q.shape[1]
  P = perm_matrix(random.permutation(n))
  W = mat.eye(n)
  for i in range(0, k):
    W *= Q * P
  return W

# SUBSTITUTE PRODUCT WITH SIGN*PERM

def signperm_kernel(deg, k):
  return mix_signperm(expand_signperm(np.mat([1]), deg, k), k)

def expand_signperm(Q, height, k):
  for i in range(0, height):
    Q = entangle_signperm(Q, k)
  return Q

def entangle_signperm(Q, k): # Q must be an isometry
  D = mix_signperm(Q, k)
  C = mix_signperm(Q, k)
  return mat.bmat([[D, C],[C, -D]]) / math.sqrt(2.0)

def mix_signperm(Q, k):
  n = Q.shape[1]
  P = perm_matrix(random.permutation(n))
  D = np.mat(np.diag(random_binary_array(n)))
  W = mat.eye(n)
  for i in range(0, k):
    W *= Q * P * D
  return W

def perm_matrix(p): # p must be a permutation array
  m = mat.zeros((len(p), len(p)))
  for i in range(0, len(p)):
    m[p[i], i] = 1
  return m

### MAIN

from numpy.random import beta
import matplotlib.pyplot as plt

# INSIGHTS:
# ... * (Q D_rnd) = recursively symmetric
# ... * (Q D_str) ~ Gaussian unstructured
# ... * (Q D_str P) ~ recursively symmetric
# ... * (Q P) < recursively symmetric

def expt(name, test_samples, K, subselect_count):
	K = subselect(K, subselect_count)
	return (name, [kernel_dot_error(K, s[0], s[1]).item((0,0)) for s in test_samples])

deg = 10
dim = 2**deg
prod_count = 7
subselect_count = 21
common_samples = samples(dim, 10000)

expts = [
	expt("recursively symmetric signs", common_samples, recursive_kernel_with_knives(deg), subselect_count),
	expt("recursively symmetric", common_samples, recursive_kernel(deg, prod_count), subselect_count),
	expt("gaussian unstructured", common_samples, gaussian_kernel(deg), subselect_count),
	expt("gaussian orthogonal", common_samples, gaussian_orthogonal_kernel(deg), subselect_count),
	# expt("permutation product", common_samples, perm_kernel(deg, prod_count), subselect_count),
	# expt("signed permutation product", common_samples, signperm_kernel(deg, prod_count), subselect_count),
]
plt.style.use('bmh')
fig, ax = plt.subplots()
for e in expts:
	ax.hist(e[1], histtype="stepfilled", bins=40, alpha=0.6, normed=True, label=e[0])
ax.set_title("kernel dot product error distribution")
ax.legend(prop={'size': 10})
plt.show()
