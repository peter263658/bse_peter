import torch
import torch.nn as nn


# class Stft(nn.Module):
#     def __init__(self, n_dft=1024, hop_size=512, win_length=None,
#                  onesided=True, is_complex=True):

#         super().__init__()

#         self.n_dft = n_dft
#         self.hop_size = hop_size
#         self.win_length = n_dft if win_length is None else win_length
#         self.onesided = onesided
#         self.is_complex = is_complex

#     # def forward(self, x: torch.Tensor):
#     #     "Expected input has shape (batch_size, n_channels, time_steps)"

#     #     window = torch.hann_window(self.win_length, device=x.device)

#     #     y = torch.stft(x, self.n_dft, hop_length=self.hop_size,
#     #                    win_length=self.win_length, onesided=self.onesided,
#     #                    return_complex=True, window=window, normalized=True)
        
#     #     y = y[:, 1:] # Remove DC component (f=0hz)

#     #     # y.shape == (batch_size*channels, time, freqs)

#     #     if not self.is_complex:
#     #         y = torch.view_as_real(y)
#     #         y = y.movedim(-1, 1) # move complex dim to front

#     #     return y
#     def forward(self, x: torch.Tensor):
#             """
#             x shape: (batch, channels, time) 或 (batch, time)
#             return: (batch*channels, frames, freqs-1) [complex]
#             """
#             # --- 1. reshape & dtype ---
#             if x.ndim == 3:                       # (B, C, T) → (B*C, T)
#                 b, c, t = x.shape
#                 x = x.reshape(b * c, t)
#             x = x.to(torch.float32)               # 避免 float64/32 不匹配

#             # --- 2. 執行 STFT ---
#             window = torch.hann_window(self.win_length, dtype=x.dtype,
#                                     device=x.device)
#             y = torch.stft(
#                 x, self.n_dft,
#                 hop_length=self.hop_size,
#                 win_length=self.win_length,
#                 window=window,
#                 onesided=self.onesided,
#                 normalized=True,
#                 return_complex=True,
#             )

#             # --- 3. 後處理 ---
#             y = y[:, 1:, :]                       # 去掉 DC (f=0 Hz)

#             if not self.is_complex:
#                 y = torch.view_as_real(y)         # 轉成 (real, imag) 兩個 channel
#                 y = y.movedim(-1, 1)              # 把 complex 維放到 dim=1

#             return y


class Stft(nn.Module):
    def __init__(self, n_dft=1024, hop_size=512, win_length=None,
                 onesided=True, is_complex=True):
        super().__init__()
        self.n_dft = n_dft
        self.hop_size = hop_size
        self.win_length = n_dft if win_length is None else win_length
        self.onesided = onesided
        self.is_complex = is_complex

    def forward(self, x: torch.Tensor):
        """
        x shape: (batch, channels, time) or (batch, time)
        return: (batch*channels, frames, freqs-1) [complex]
        """
        # --- 1. reshape & dtype ---
        original_shape = x.shape
        if x.ndim == 3:                       # (B, C, T) → (B*C, T)
            b, c, t = x.shape
            x = x.reshape(b * c, t)
        x = x.to(torch.float32)               # Avoid float64/32 mismatch

        # --- 2. Execute STFT ---
        window = torch.hann_window(self.win_length, dtype=x.dtype,
                                device=x.device)
        y = torch.stft(
            x, self.n_dft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=window,
            onesided=self.onesided,
            normalized=True,
            return_complex=True,
        )

        # --- 3. Post-processing ---
        y = y[:, 1:, :]                       # Remove DC (f=0 Hz)

        # --- 5. Convert to real representation if needed ---
        if not self.is_complex:
            y = torch.view_as_real(y)         # Convert to (real, imag) channels
            y = y.movedim(-1, 1)              # Move complex dim to dim=1

        return y

class IStft(Stft):

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels=freq_bins, time_steps)"
        window = torch.hann_window(self.win_length, device=x.device)

        y = torch.istft(x, self.n_dft, hop_length=self.hop_size,
                        win_length=self.win_length, onesided=self.onesided,
                        window=window,normalized=True)

        return y
