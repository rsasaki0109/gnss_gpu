#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/interference.h"
#include <cufft.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

namespace gnss_gpu {

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

__global__ void apply_hann_window_kernel(const float* signal, float* output,
                                         int fft_size, int hop_size, int n_frames) {
    int frame = blockIdx.x;
    int tid = threadIdx.x;
    if (frame >= n_frames) return;

    for (int i = tid; i < fft_size; i += blockDim.x) {
        int sample_idx = frame * hop_size + i;
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (fft_size - 1)));
        output[frame * fft_size + i] = signal[sample_idx] * window;
    }
}

__global__ void power_spectrum_kernel(const cufftComplex* fft_out, float* power_db,
                                      int n_bins, int n_frames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * n_bins;
    if (idx >= total) return;

    float re = fft_out[idx].x;
    float im = fft_out[idx].y;
    float eps = 1e-20f;
    power_db[idx] = 10.0f * log10f(re * re + im * im + eps);
}

// Compute noise floor for each bin using median of neighboring bins' mean power.
// This estimates the local spectral background, so narrowband interference
// (which affects only a few bins) stands out above the floor.
__global__ void noise_floor_kernel(const float* spectrogram, float* noise_floor,
                                   const float* bin_mean_power,
                                   int n_frames, int n_bins) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= n_bins) return;

    // Use a sliding window of neighboring bins to estimate local noise floor.
    // Exclude the center region (+/- guard bins) and take the median of the rest.
    const int half_window = 16;  // look at 32 neighboring bins
    const int guard = 2;         // exclude +/- 2 bins around center

    // Collect neighbor values (excluding guard band around current bin)
    float neighbors[64];
    int count = 0;
    for (int b = bin - half_window; b <= bin + half_window; b++) {
        if (b < 0 || b >= n_bins) continue;
        if (b >= bin - guard && b <= bin + guard) continue;
        if (count < 64) {
            neighbors[count++] = bin_mean_power[b];
        }
    }

    // Simple selection sort to find median (small array)
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (neighbors[j] < neighbors[i]) {
                float tmp = neighbors[i];
                neighbors[i] = neighbors[j];
                neighbors[j] = tmp;
            }
        }
    }

    noise_floor[bin] = (count > 0) ? neighbors[count / 2] : bin_mean_power[bin];
}

// Helper kernel: compute mean power per bin across all time frames
__global__ void bin_mean_power_kernel(const float* spectrogram, float* bin_mean,
                                      int n_frames, int n_bins) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= n_bins) return;

    float sum = 0.0f;
    for (int f = 0; f < n_frames; f++) {
        sum += spectrogram[f * n_bins + bin];
    }
    bin_mean[bin] = sum / n_frames;
}

__global__ void detect_peaks_kernel(const float* spectrogram, const float* noise_floor,
                                    float threshold_db, int* peak_mask,
                                    int n_frames, int n_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * n_bins;
    if (idx >= total) return;

    int bin = idx % n_bins;
    peak_mask[idx] = (spectrogram[idx] - noise_floor[bin] > threshold_db) ? 1 : 0;
}

__global__ void notch_excise_kernel(cufftComplex* fft_data, const int* peak_mask,
                                    int n_bins, int n_frames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * n_bins;
    if (idx >= total) return;

    if (peak_mask[idx]) {
        fft_data[idx].x = 0.0f;
        fft_data[idx].y = 0.0f;
    }
}

__global__ void overlap_add_kernel(const float* frames, float* output,
                                   int hop_size, int n_frames, int fft_size,
                                   int n_samples) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= n_samples) return;

    float sum = 0.0f;
    float window_sum = 0.0f;

    // Find all frames that overlap this sample
    int first_frame = max(0, (sample - fft_size + 1 + hop_size - 1) / hop_size);
    int last_frame = min(n_frames - 1, sample / hop_size);

    for (int f = first_frame; f <= last_frame; f++) {
        int i = sample - f * hop_size;
        if (i >= 0 && i < fft_size) {
            float window = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (fft_size - 1)));
            sum += frames[f * fft_size + i] * window;
            window_sum += window * window;
        }
    }

    output[sample] = (window_sum > 1e-8f) ? sum / window_sum : 0.0f;
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------

void compute_stft(const float* signal, float* spectrogram,
                  int n_samples, int fft_size, int hop_size, double sampling_freq) {
    int n_frames = (n_samples - fft_size) / hop_size + 1;
    if (n_frames <= 0) return;

    int n_bins = fft_size / 2 + 1;

    // Allocate device memory
    float *d_signal, *d_windowed;
    cufftComplex *d_fft_out;
    float *d_power;

    CUDA_CHECK(cudaMalloc(&d_signal, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_windowed, (size_t)n_frames * fft_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fft_out, (size_t)n_frames * n_bins * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_power, (size_t)n_frames * n_bins * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_signal, signal, n_samples * sizeof(float), cudaMemcpyHostToDevice));

    // Apply Hann window to each frame
    int block = 256;
    apply_hann_window_kernel<<<n_frames, block>>>(d_signal, d_windowed,
                                                   fft_size, hop_size, n_frames);

    // Batch R2C FFT
    cufftHandle plan;
    int n[1] = {fft_size};
    cufftPlanMany(&plan, 1, n,
                  nullptr, 1, fft_size,    // input layout
                  nullptr, 1, n_bins,      // output layout
                  CUFFT_R2C, n_frames);
    cufftExecR2C(plan, d_windowed, d_fft_out);
    cufftDestroy(plan);

    // Compute power spectrum in dB
    int total = n_frames * n_bins;
    int grid = (total + block - 1) / block;
    power_spectrum_kernel<<<grid, block>>>(d_fft_out, d_power, n_bins, n_frames);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(spectrogram, d_power, (size_t)n_frames * n_bins * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_windowed));
    CUDA_CHECK(cudaFree(d_fft_out));
    CUDA_CHECK(cudaFree(d_power));
}

int detect_interference(const float* spectrogram, InterferenceDetection* detections,
                        int max_detections, int n_frames, int fft_size,
                        double sampling_freq, float threshold_db) {
    int n_bins = fft_size / 2 + 1;

    // Compute noise floor and peak mask on GPU
    float *d_spec, *d_noise, *d_bin_mean;
    int *d_mask;

    CUDA_CHECK(cudaMalloc(&d_spec, (size_t)n_frames * n_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bin_mean, n_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_noise, n_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, (size_t)n_frames * n_bins * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_spec, spectrogram, (size_t)n_frames * n_bins * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid_bins = (n_bins + block - 1) / block;

    // Step 1: compute mean power per bin across time frames
    bin_mean_power_kernel<<<grid_bins, block>>>(d_spec, d_bin_mean, n_frames, n_bins);

    // Step 2: compute noise floor using median of neighboring bins' mean power
    noise_floor_kernel<<<grid_bins, block>>>(d_spec, d_noise, d_bin_mean,
                                              n_frames, n_bins);

    int total = n_frames * n_bins;
    int grid_total = (total + block - 1) / block;
    detect_peaks_kernel<<<grid_total, block>>>(d_spec, d_noise, threshold_db,
                                               d_mask, n_frames, n_bins);

    // Copy results to host for classification
    std::vector<float> noise_floor(n_bins);
    std::vector<int> mask(total);
    CUDA_CHECK(cudaMemcpy(noise_floor.data(), d_noise, n_bins * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mask.data(), d_mask, total * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_spec));
    CUDA_CHECK(cudaFree(d_bin_mean));
    CUDA_CHECK(cudaFree(d_noise));
    CUDA_CHECK(cudaFree(d_mask));

    // Classify interference: group adjacent frequency bins with peaks
    // Count per-bin detections across frames
    std::vector<int> bin_count(n_bins, 0);
    for (int f = 0; f < n_frames; f++) {
        for (int b = 0; b < n_bins; b++) {
            bin_count[b] += mask[f * n_bins + b];
        }
    }

    // Find contiguous frequency regions with significant detections
    int n_det = 0;
    double freq_res = sampling_freq / fft_size;
    int min_frame_count = std::max(1, n_frames / 4);

    int b = 0;
    while (b < n_bins && n_det < max_detections) {
        if (bin_count[b] >= min_frame_count) {
            int start_bin = b;
            float max_power = -1e30f;
            int max_bin = b;

            while (b < n_bins && bin_count[b] >= min_frame_count) {
                // Find peak power in this bin across all frames
                for (int f = 0; f < n_frames; f++) {
                    float p = spectrogram[f * n_bins + b];
                    if (p > max_power) {
                        max_power = p;
                        max_bin = b;
                    }
                }
                b++;
            }
            int end_bin = b;
            int bw_bins = end_bin - start_bin;

            // Find temporal extent
            int first_frame = n_frames, last_frame = 0;
            for (int f = 0; f < n_frames; f++) {
                for (int bb = start_bin; bb < end_bin; bb++) {
                    if (mask[f * n_bins + bb]) {
                        first_frame = std::min(first_frame, f);
                        last_frame = std::max(last_frame, f);
                    }
                }
            }

            InterferenceDetection det;
            det.center_freq_hz = static_cast<float>((start_bin + end_bin) * 0.5 * freq_res);
            det.bandwidth_hz = static_cast<float>(bw_bins * freq_res);
            det.power_db = max_power - noise_floor[max_bin];
            det.start_frame = first_frame;
            det.end_frame = last_frame;

            // Classify based on bandwidth and temporal pattern
            float duty = static_cast<float>(last_frame - first_frame + 1) / n_frames;

            if (bw_bins <= 2) {
                det.type = InterferenceType::CW;
            } else if (bw_bins <= fft_size / 8) {
                // Check for chirp: does peak bin shift across frames?
                bool is_chirp = false;
                if (n_frames >= 4) {
                    int first_q_peak = 0, last_q_peak = 0;
                    int q_frames = n_frames / 4;
                    // Peak bin in first quarter
                    float fp = -1e30f;
                    for (int bb = start_bin; bb < end_bin; bb++) {
                        float s = 0;
                        for (int f = 0; f < q_frames; f++) s += spectrogram[f * n_bins + bb];
                        if (s > fp) { fp = s; first_q_peak = bb; }
                    }
                    // Peak bin in last quarter
                    fp = -1e30f;
                    for (int bb = start_bin; bb < end_bin; bb++) {
                        float s = 0;
                        for (int f = n_frames - q_frames; f < n_frames; f++)
                            s += spectrogram[f * n_bins + bb];
                        if (s > fp) { fp = s; last_q_peak = bb; }
                    }
                    if (abs(first_q_peak - last_q_peak) >= 2) is_chirp = true;
                }
                det.type = is_chirp ? InterferenceType::CHIRP : InterferenceType::NARROWBAND;
            } else {
                det.type = (duty < 0.5f) ? InterferenceType::PULSED : InterferenceType::NARROWBAND;
            }

            detections[n_det++] = det;
        } else {
            b++;
        }
    }

    return n_det;
}

void excise_interference(const float* input, float* output,
                        const float* spectrogram, int n_samples,
                        int fft_size, int hop_size, float threshold_db) {
    int n_frames = (n_samples - fft_size) / hop_size + 1;
    if (n_frames <= 0) {
        memcpy(output, input, n_samples * sizeof(float));
        return;
    }
    int n_bins = fft_size / 2 + 1;

    // Device memory
    float *d_signal, *d_windowed, *d_output, *d_spec, *d_noise, *d_bin_mean, *d_reconstructed;
    cufftComplex *d_fft;
    int *d_mask;

    CUDA_CHECK(cudaMalloc(&d_signal, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_windowed, (size_t)n_frames * fft_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fft, (size_t)n_frames * n_bins * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_spec, (size_t)n_frames * n_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bin_mean, n_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_noise, n_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, (size_t)n_frames * n_bins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_reconstructed, (size_t)n_frames * fft_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n_samples * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_signal, input, n_samples * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spec, spectrogram, (size_t)n_frames * n_bins * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;

    // Step 1: Window input frames
    apply_hann_window_kernel<<<n_frames, block>>>(d_signal, d_windowed,
                                                   fft_size, hop_size, n_frames);

    // Step 2: Forward FFT
    cufftHandle plan_fwd;
    int n_arr[1] = {fft_size};
    cufftPlanMany(&plan_fwd, 1, n_arr,
                  nullptr, 1, fft_size,
                  nullptr, 1, n_bins,
                  CUFFT_R2C, n_frames);
    cufftExecR2C(plan_fwd, d_windowed, d_fft);
    cufftDestroy(plan_fwd);

    // Step 3: Compute noise floor and peak mask from spectrogram
    int grid_bins = (n_bins + block - 1) / block;
    bin_mean_power_kernel<<<grid_bins, block>>>(d_spec, d_bin_mean, n_frames, n_bins);
    noise_floor_kernel<<<grid_bins, block>>>(d_spec, d_noise, d_bin_mean,
                                              n_frames, n_bins);

    int total = n_frames * n_bins;
    int grid_total = (total + block - 1) / block;
    detect_peaks_kernel<<<grid_total, block>>>(d_spec, d_noise, threshold_db,
                                               d_mask, n_frames, n_bins);

    // Step 4: Notch excision - zero out masked bins
    notch_excise_kernel<<<grid_total, block>>>(d_fft, d_mask, n_bins, n_frames);

    // Step 5: Inverse FFT
    cufftHandle plan_inv;
    cufftPlanMany(&plan_inv, 1, n_arr,
                  nullptr, 1, n_bins,
                  nullptr, 1, fft_size,
                  CUFFT_C2R, n_frames);
    cufftExecC2R(plan_inv, d_fft, d_reconstructed);
    cufftDestroy(plan_inv);

    // Step 6: Scale by 1/fft_size (cuFFT does unnormalized transforms)
    // and overlap-add reconstruction
    // First normalize the IFFT output (done inline in overlap_add via window)
    // cuFFT C2R output is scaled by fft_size, so we need to account for that
    // We handle normalization in overlap_add_kernel implicitly via window division

    // Zero the output buffer
    CUDA_CHECK(cudaMemset(d_output, 0, n_samples * sizeof(float)));

    int grid_samples = (n_samples + block - 1) / block;
    overlap_add_kernel<<<grid_samples, block>>>(d_reconstructed, d_output,
                                                 hop_size, n_frames, fft_size,
                                                 n_samples);

    // Scale by 1/fft_size for cuFFT normalization
    // Simple scaling kernel inline via thrust or manual
    // We'll copy to host and scale there for simplicity
    std::vector<float> h_output(n_samples);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_samples * sizeof(float), cudaMemcpyDeviceToHost));
    float scale = 1.0f / fft_size;
    for (int i = 0; i < n_samples; i++) {
        output[i] = h_output[i] * scale;
    }

    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_windowed));
    CUDA_CHECK(cudaFree(d_fft));
    CUDA_CHECK(cudaFree(d_spec));
    CUDA_CHECK(cudaFree(d_bin_mean));
    CUDA_CHECK(cudaFree(d_noise));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_reconstructed));
    CUDA_CHECK(cudaFree(d_output));
}

}  // namespace gnss_gpu
