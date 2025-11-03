import numpy as np
from numpy.core.multiarray import ndarray


def generate_blue_noise_fft(size, low_freq=10, band_width=30):
    """
    Generate a blue noise mask using FFT spectral shaping.

    Args:
        size (int): Width and height of square output image.
        low_freq (int): Inner radius of frequency band (low cutoff).
        high_freq (int): Outer radius of frequency band (high cutoff).

    Returns:
        2D np.uint8 array: Blue noise threshold map normalized to 0-255.
    """

    high_freq = low_freq + band_width
    # 1. Generate white noise in frequency domain (complex)
    noise = np.random.randn(size, size) + 1j * np.random.randn(size, size)

    # 2. Construct frequency grid centered at zero frequency
    u = np.fft.fftfreq(size) * size
    v = np.fft.fftfreq(size) * size
    U, V = np.meshgrid(u, v)
    radius = np.sqrt(U ** 2 + V ** 2)

    # 3. Create band-pass mask: allow frequencies between low_freq and high_freq
    band_mask = (radius >= low_freq) & (radius <= high_freq)

    # 4. Apply mask: keep only frequencies in the blue noise band
    noise_filtered = noise * band_mask

    # 5. Enforce Hermitian symmetry to get a real spatial pattern after iFFT
    def hermitian_symmetrize(arr):
        size = arr.shape[0]
        arr_sym = np.zeros_like(arr, dtype=np.complex128)

        # Copy DC and Nyquist components
        arr_sym[0, 0] = arr[0, 0]
        arr_sym[size // 2, 0] = arr[size // 2, 0]
        arr_sym[0, size // 2] = arr[0, size // 2]
        arr_sym[size // 2, size // 2] = arr[size // 2, size // 2]

        # Copy positive frequencies
        for i in range(1, size // 2):
            for j in range(size):
                arr_sym[i, j] = arr[i, j]
                arr_sym[size - i, (size - j) % size] = np.conj(arr[i, j])

        return arr_sym

    noise_sym = hermitian_symmetrize(noise_filtered)

    # 6. Inverse FFT to get spatial pattern
    blue_noise = np.fft.ifft2(noise_sym).real

    blue_noise = (blue_noise - blue_noise.mean()) / blue_noise.std()

    return blue_noise

def generate_blue_noise_fft_multichannel(shape, low_freq=10, band_width=20):
    assert len(shape) == 3, "shape needs to be 3 dimentional"
    H,W,C = shape
    size = max(H,W)
    noise_list = [generate_blue_noise_fft(size,low_freq,band_width) for c in range(C)]
    noise_array = np.stack(noise_list,axis=2)[:H,:W]
    return noise_array


def create_bayer_matrix(level):
    """Create Bayer matrix of size 2^level x 2^level normalized to [-1,1]."""
    if level < 1:
        raise ValueError("Level must be >= 1")
    base = np.array([[0, 2],
                     [3, 1]])

    def recursive_bayer(n):
        if n == 1:
            return base
        else:
            smaller = recursive_bayer(n - 1)
            return np.block([
                [4 * smaller, 4 * smaller + 2],
                [4 * smaller + 3, 4 * smaller + 1]
            ])

    B = recursive_bayer(level)
    B = B.astype(np.float32)
    # B = (B / B.max()) * 2 - 1  # Normalize to [-1, 1]
    return B


def generate_bayer_matrices(level: int, shape=None) -> ndarray:
    assert len(shape) ==3, "shape needs to be 3 dimentional"
    base = create_bayer_matrix(level)
    base = np.tile(base, (shape[0] // base.shape[0] + 1, shape[1] // base.shape[1] + 1))
    B_list = []
    for c in range(shape[2]):
        shift_x, shift_y = c, c  # Can be improved later
        shifted = np.roll(base, shift=(shift_y, shift_x), axis=(0, 1))
        B_list.append(shifted[:shape[0], :shape[1]])
    return np.stack(B_list,axis=2)



def tile_matrix(mat, H, W):
    """Tiles matrix to match (H, W) dimensions."""
    h, w = mat.shape
    return np.tile(mat, (H // h + 1, W // w + 1))[:H, :W]

PATTERN_REGISTRY ={
    'blue_noise':{"func" : generate_blue_noise_fft_multichannel,
     "params": {
         "low_freq": {'type':'range', 'values': [0,500]}, #TODO: change to image size limit somehow
         "band_width": {'type':'range', 'values': [0,500]}
     }},
    'bayer':{"func": generate_bayer_matrices,
     "params": {
         "level": {'type': 'list', 'values': [1, 2, 3, 4]},  # TODO: change to image size limit somehow
     }}
}


def generate_pattern(pattern_name, shape, pattern_params):
    return PATTERN_REGISTRY[pattern_name]['func'](shape=shape,**pattern_params)
