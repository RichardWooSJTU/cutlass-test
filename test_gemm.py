import numpy as np 
import cutlass

m = 40
k = 1024
n = 8192

a = np.random.randint(-127, 127, [m, k], 'int8')
b_t = np.random.randint(-127, 127, [n, k], 'int8')

# a = np.ones([m, k]).astype('int8')
# b_t = np.ones([n, k]).astype('int8')

b = b_t.transpose((1, 0))

scale = np.random.rand(n).astype('float32')
# scale = np.array([0.1] * n).astype('float32')

print(a)
print(b)
print(b_t)

c = cutlass.gemm_dequant(a, b_t, scale)
# c_2 = cutlass.gemm_dequant(a, b, scale)

c_ref = np.dot(a.astype('int32'), b.astype('int32'))
c_ref = c_ref * scale 
print(c)
# print(c_2)
print(c_ref)
print(c_ref.shape)


# c_ref_2 = np.dot(a.astype('int32'), test_b.astype('int32'))
# c_ref_2 = c_ref_2 * scale 
# print(c_ref_2)

diff = np.allclose(c, c_ref)
print(diff)

