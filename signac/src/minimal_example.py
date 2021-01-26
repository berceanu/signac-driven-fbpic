import numpy as np

np.random.seed(42)
H = np.random.uniform(0.1, 1.0, size=(6, 6))
r, c = H.shape

print(H)
print()

x = np.linspace(0, 10, c)

H_masked = np.ma.masked_all_like(H)

mask_1d = H.max(axis=1) > 0.95
mask_2d = H < 0.5

H_masked[mask_1d] = H[mask_1d]
H_masked.mask = np.ma.mask_or(np.ma.getmask(H_masked), mask_2d)

print(H_masked)
print()

wa = (np.sum(x * H_masked, axis=1)) / np.sum(H_masked, axis=1)
print(wa)

new_x = np.ma.masked_where(np.ma.getmask(wa), x)
print(new_x)
