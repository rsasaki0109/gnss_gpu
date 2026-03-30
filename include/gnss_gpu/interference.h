#pragma once

namespace gnss_gpu {

enum class InterferenceType { NONE = 0, CW, NARROWBAND, PULSED, CHIRP };

struct InterferenceDetection {
    InterferenceType type;
    float center_freq_hz;
    float bandwidth_hz;
    float power_db;
    int start_frame;
    int end_frame;
};

// Compute Short-Time Fourier Transform power spectrogram (dB)
// signal: [n_samples] input signal
// spectrogram: [n_frames * (fft_size/2+1)] output power spectrogram in dB
// n_samples: number of input samples
// fft_size: FFT window size
// hop_size: hop between frames
// sampling_freq: sampling frequency in Hz
void compute_stft(const float* signal, float* spectrogram,
                  int n_samples, int fft_size, int hop_size, double sampling_freq);

// Detect interference in spectrogram
// spectrogram: [n_frames * (fft_size/2+1)] power spectrogram in dB
// detections: output array of detected interference
// max_detections: maximum number of detections to return
// n_frames: number of STFT frames
// fft_size: FFT size used
// sampling_freq: sampling frequency in Hz
// threshold_db: detection threshold above noise floor in dB
// returns number of detections found
int detect_interference(const float* spectrogram, InterferenceDetection* detections,
                        int max_detections, int n_frames, int fft_size,
                        double sampling_freq, float threshold_db);

// Excise interference from signal using spectral notch filtering
// input: [n_samples] input signal
// output: [n_samples] cleaned output signal
// spectrogram: [n_frames * (fft_size/2+1)] power spectrogram in dB
// n_samples: number of samples
// fft_size: FFT size
// hop_size: hop size
// threshold_db: excision threshold above noise floor in dB
void excise_interference(const float* input, float* output,
                        const float* spectrogram, int n_samples,
                        int fft_size, int hop_size, float threshold_db);

}  // namespace gnss_gpu
