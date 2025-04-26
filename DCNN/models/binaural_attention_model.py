import torch

from DCNN.utils.apply_mask import apply_mask

from .model import DCNN




# class BinauralAttentionDCNN(DCNN):

#     def forward(self, inputs):
#         # Properly indented function body
#         cspecs_l = self.stft(inputs[:, 0])
#         cspecs_r = self.stft(inputs[:, 1])
#         cspecs = torch.stack((cspecs_l, cspecs_r), dim=1)

#         encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
#         encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))

#         attention_in = torch.cat((encoder_out_l[-1], encoder_out_r[-1]), dim=1)
        
#         x_attn = self.mattn(attention_in)
#         x_l_mattn = x_attn[:,:128,:,:]
#         x_r_mattn = x_attn[:,128:,:,:]
        
#         x_l = self.decoder(x_l_mattn, encoder_out_l)
#         x_r = self.decoder(x_r_mattn, encoder_out_r)

#         # Apply mask
#         out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
#         out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)

#         # Invert STFT
#         out_wav_l = self.istft(out_spec_l)
#         out_wav_r = self.istft(out_spec_r)
        
#         out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
       
#         return out_wav
class BinauralAttentionDCNN(DCNN):
    def forward(self, inputs):
        # Process left and right channels through STFT
        cspecs_l = self.stft(inputs[:, 0])
        cspecs_r = self.stft(inputs[:, 1])
        
        # Ensure both channels have the same time frames
        min_frames = min(cspecs_l.shape[-1], cspecs_r.shape[-1])
        if cspecs_l.shape[-1] != cspecs_r.shape[-1]:
            print(f"Warning: Truncating STFT outputs to match: {cspecs_l.shape[-1]} vs {cspecs_r.shape[-1]}")
            cspecs_l = cspecs_l[..., :min_frames]
            cspecs_r = cspecs_r[..., :min_frames]
        
        # Continue with the rest of your model...
        encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
        encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))
        
        # Ensure encoder outputs have matching dimensions
        for i in range(len(encoder_out_l)):
            min_frames = min(encoder_out_l[i].shape[-1], encoder_out_r[i].shape[-1])
            if encoder_out_l[i].shape[-1] != encoder_out_r[i].shape[-1]:
                print(f"Truncating encoder layer {i}: {encoder_out_l[i].shape[-1]} vs {encoder_out_r[i].shape[-1]}")
                encoder_out_l[i] = encoder_out_l[i][..., :min_frames]
                encoder_out_r[i] = encoder_out_r[i][..., :min_frames]
        
        # Continue with your existing code...
        attention_in = torch.cat((encoder_out_l[-1], encoder_out_r[-1]), dim=1)
        
        x_attn = self.mattn(attention_in)
        x_l_mattn = x_attn[:, :128, :, :]
        x_r_mattn = x_attn[:, 128:, :, :]
        
        x_l = self.decoder(x_l_mattn, encoder_out_l)
        x_r = self.decoder(x_r_mattn, encoder_out_r)

        # Apply mask
        out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
        out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)

        # Invert STFT
        out_wav_l = self.istft(out_spec_l)
        out_wav_r = self.istft(out_spec_r)
        
        out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
    
        return out_wav