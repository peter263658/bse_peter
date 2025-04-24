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

    # def forward(self, x: torch.Tensor):
    #     "Expected input has shape (batch_size, n_channels, time_steps)"

    #     window = torch.hann_window(self.win_length, device=x.device)

    #     y = torch.stft(x, self.n_dft, hop_length=self.hop_size,
    #                    win_length=self.win_length, onesided=self.onesided,
    #                    return_complex=True, window=window, normalized=True)
        
    #     y = y[:, 1:] # Remove DC component (f=0hz)

    #     # y.shape == (batch_size*channels, time, freqs)

    #     if not self.is_complex:
    #         y = torch.view_as_real(y)
    #         y = y.movedim(-1, 1) # move complex dim to front

    #     return y


class Stft(nn.Module):
    def __init__(self, n_dft=1024, hop_size=512, win_length=None,
                 onesided=True, is_complex=True, remove_dc=False):
        super().__init__()
        self.n_dft = n_dft
        self.hop_size = hop_size
        self.win_length = n_dft if win_length is None else win_length
        self.onesided = onesided
        self.is_complex = is_complex
        self.remove_dc = True
        # self.remove_dc = False

    def forward(self, x: torch.Tensor):
        """Expected input has shape (batch_size, n_channels, time_steps) or (batch_size, time_steps) or (time_steps)"""
        
        # Handle 1D input (just time samples)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions (1, 1, T)
            print("Warning: Input to STFT is 1D, adding batch and channel dimensions")
        # Handle 2D input (batch, time)
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
            print("Warning: Input to STFT missing channel dimension, assuming mono")
        
        # Process each channel separately
        batch_size, n_channels, time_steps = x.shape
        y_list = []
        
        for c in range(n_channels):
            channel_data = x[:, c, :]
            window = torch.hann_window(self.win_length, device=channel_data.device)
            y_c = torch.stft(channel_data, self.n_dft, hop_length=self.hop_size,
                        win_length=self.win_length, onesided=self.onesided,
                        return_complex=True, window=window, normalized=True)
            
            # Don't remove DC component here - we'll do it after stacking if needed
            y_list.append(y_c)
        
        # For multi-channel, stack along new channel dimension
        if n_channels > 1:
            y = torch.stack(y_list, dim=1)  # Shape: (batch, channels, freq, time)
        else:
            y = y_list[0]  # Shape: (batch, freq, time)
        
        # Optionally remove DC component after stacking
        if self.remove_dc:
            if n_channels > 1:  # Multi-channel case
                y = y[:, :, 1:, :]  # Remove first frequency bin (DC)
            else:  # Single-channel case
                y = y[:, 1:, :]  # Remove first frequency bin (DC)
        
        return y


class IStft(Stft):

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels=freq_bins, time_steps)"
        window = torch.hann_window(self.win_length, device=x.device)

        y = torch.istft(x, self.n_dft, hop_length=self.hop_size,
                        win_length=self.win_length, onesided=self.onesided,
                        window=window,normalized=True)

        return y

