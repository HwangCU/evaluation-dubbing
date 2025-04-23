# renderer.py
import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioRenderer:
    """Class for rendering final audio with background noise and reverberation."""
    
    def __init__(self, use_source_separation: bool = True, add_reverberation: bool = True):
        """
        Initialize the audio renderer.
        
        Args:
            use_source_separation: Whether to extract background noise from original audio
            add_reverberation: Whether to add reverberation matching the original audio
        """
        self.use_source_separation = use_source_separation
        self.add_reverberation = add_reverberation
        
        logger.info(f"Initializing audio renderer: use_source_separation={use_source_separation}, add_reverberation={add_reverberation}")
    
    def render(
        self,
        src_audio_path: str,
        tts_audio_paths: List[str],
        segment_timings: List[Tuple[float, float]],
        output_path: str
    ) -> str:
        """
        Render the final dubbed audio by combining TTS with background from original.
        
        Args:
            src_audio_path: Path to the source audio file
            tts_audio_paths: List of paths to TTS audio files for each segment
            segment_timings: List of (start_time, end_time) tuples for each segment
            output_path: Path to save the output audio
        
        Returns:
            Path to the rendered audio
        """
        try:
            import librosa
            import soundfile as sf
            
            logger.info(f"Rendering dubbed audio to {output_path}")
            
            # Load source audio
            src_audio, src_sr = librosa.load(src_audio_path, sr=None)
            
            # Create output audio array (same length as source)
            output_audio = np.zeros_like(src_audio)
            
            # Extract background noise if enabled
            if self.use_source_separation:
                background = self._extract_background(src_audio_path, src_audio, src_sr)
            else:
                # Use low-volume version of original as background
                background = src_audio * 0.1
            
            # Add background to output
            output_audio += background
            
            # Process each TTS segment
            for i, (tts_path, (start_time, end_time)) in enumerate(zip(tts_audio_paths, segment_timings)):
                # Load TTS audio
                tts_audio, tts_sr = librosa.load(tts_path, sr=src_sr)
                
                # Apply reverberation if enabled
                if self.add_reverberation:
                    tts_audio = self._add_reverberation(tts_audio, src_audio, src_sr)
                
                # Calculate start and end samples
                start_sample = int(start_time * src_sr)
                end_sample = int(end_time * src_sr)
                
                # Check if TTS audio fits in the segment
                tts_duration = len(tts_audio) / tts_sr
                segment_duration = (end_sample - start_sample) / src_sr
                
                if tts_duration > segment_duration * 1.1:
                    # TTS audio is too long, need to compress it
                    logger.warning(f"TTS audio for segment {i} is too long ({tts_duration:.2f}s > {segment_duration:.2f}s), compressing")
                    tts_audio = self._time_compress(tts_audio, tts_sr, segment_duration)
                
                # Place TTS audio in the output at the correct position
                end_pos = min(start_sample + len(tts_audio), len(output_audio))
                output_audio[start_sample:end_pos] = tts_audio[:end_pos - start_sample]
            
            # Normalize audio
            output_audio = self._normalize_audio(output_audio)
            
            # Save output audio
            sf.write(output_path, output_audio, src_sr)
            
            logger.info(f"Dubbed audio successfully rendered to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error rendering audio: {e}")
            
            # Create a fallback audio file
            self._create_fallback_audio(output_path)
            return output_path
    
    def _extract_background(self, src_audio_path: str, src_audio: np.ndarray, src_sr: int) -> np.ndarray:
        """
        Extract background noise from source audio using source separation.
        
        Args:
            src_audio_path: Path to the source audio file
            src_audio: Source audio array
            src_sr: Source audio sample rate
        
        Returns:
            Background noise audio array
        """
        try:
            # Try to use advanced source separation methods
            from asteroid.models import BaseModel
            
            # Try to load a pre-trained model
            model = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
            
            # Convert source audio to expected format
            if src_sr != 8000:
                from librosa.core import resample
                src_audio_8k = resample(src_audio, orig_sr=src_sr, target_sr=8000)
            else:
                src_audio_8k = src_audio
            
            # Separate audio sources
            est_sources = model.separate(torch.tensor(src_audio_8k).unsqueeze(0))
            background = est_sources[0, 1].numpy()  # Assuming background is the second source
            
            # Resample back to original rate if needed
            if src_sr != 8000:
                background = resample(background, orig_sr=8000, target_sr=src_sr)
            
            logger.info("Background extracted using advanced source separation")
            return background
            
        except Exception:
            logger.info("Advanced source separation not available, using simple method")
            
            # Simple method: use a lowpass filter to extract background
            from scipy import signal
            
            # Design a lowpass filter
            cutoff = 200  # Hz
            nyquist = src_sr / 2
            normal_cutoff = cutoff / nyquist
            b, a = signal.butter(5, normal_cutoff, btype='lowpass')
            
            # Apply filter
            background = signal.filtfilt(b, a, src_audio)
            
            # Reduce volume of original speech
            from librosa.effects import split
            
            # Find speech segments
            speech_intervals = split(src_audio, top_db=20)
            
            # Create a mask that attenuates speech regions
            mask = np.ones_like(src_audio)
            for interval in speech_intervals:
                start, end = interval
                mask[start:end] = 0.2  # Reduce volume in speech regions
            
            # Apply mask
            background = background * mask
            
            return background * 0.5  # Reduce overall volume
    
    def _add_reverberation(self, audio: np.ndarray, reference_audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Add reverberation to audio matching the reverb characteristics of reference audio.
        
        Args:
            audio: Audio to process
            reference_audio: Reference audio to match
            sr: Sample rate
        
        Returns:
            Processed audio with reverberation
        """
        try:
            from scipy import signal
            
            # Estimate reverberation time (RT60) from reference audio
            rt60 = self._estimate_rt60(reference_audio, sr)
            
            # Create a simple reverb impulse response
            decay_factor = np.exp(-6.91 * np.arange(int(rt60 * sr)) / (rt60 * sr))
            impulse_response = np.random.randn(len(decay_factor)) * decay_factor
            
            # Normalize impulse response
            impulse_response = impulse_response / np.sqrt(np.sum(impulse_response**2))
            
            # Apply convolution
            reverb_audio = signal.convolve(audio, impulse_response, mode='full')
            
            # Trim to original length
            reverb_audio = reverb_audio[:len(audio)]
            
            # Mix dry and wet signals
            dry_gain = 0.7
            wet_gain = 0.3
            
            processed_audio = dry_gain * audio + wet_gain * reverb_audio
            
            return processed_audio
            
        except Exception as e:
            logger.warning(f"Failed to add reverberation: {e}")
            return audio
    
    def _estimate_rt60(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate reverberation time (RT60) from audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
        
        Returns:
            Estimated RT60 in seconds
        """
        try:
            import numpy as np
            from scipy import signal
            
            # Compute energy decay curve
            # First, compute the energy envelope
            window_size = int(0.04 * sr)  # 40ms window
            hop_length = int(0.02 * sr)   # 20ms hop
            
            # Calculate STFT
            f, t, Zxx = signal.stft(audio, sr, nperseg=window_size, noverlap=window_size-hop_length)
            
            # Calculate energy
            energy = np.sum(np.abs(Zxx)**2, axis=0)
            
            # Convert to dB
            energy_db = 10 * np.log10(energy + 1e-10)
            
            # Normalize
            energy_db = energy_db - np.max(energy_db)
            
            # Find time when energy drops by 60dB
            threshold = -60
            
            # Find last point above threshold
            above_threshold = np.where(energy_db > threshold)[0]
            
            if len(above_threshold) > 0:
                last_point = above_threshold[-1]
                rt60 = t[last_point]
            else:
                # Default RT60 if we can't estimate
                rt60 = 0.5  # 500ms
            
            # Limit to reasonable range
            rt60 = min(max(rt60, 0.2), 2.0)
            
            logger.info(f"Estimated RT60: {rt60:.2f} seconds")
            return rt60
            
        except Exception as e:
            logger.warning(f"Failed to estimate RT60: {e}")
            return 0.5  # Default value
    
    def _time_compress(self, audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
        """
        Compress audio to fit target duration.
        
        Args:
            audio: Audio to compress
            sr: Sample rate
            target_duration: Target duration in seconds
        
        Returns:
            Compressed audio
        """
        try:
            import pyrubberband as pyrb
            
            # Calculate current duration
            current_duration = len(audio) / sr
            
            # Calculate stretch factor
            stretch_factor = target_duration / current_duration
            
            # Apply time stretching
            compressed_audio = pyrb.time_stretch(audio, sr, stretch_factor)
            
            return compressed_audio
            
        except Exception as e:
            logger.warning(f"Failed to compress audio: {e}")
            
            # Simple resampling as fallback
            target_length = int(target_duration * sr)
            from scipy import signal
            return signal.resample(audio, target_length)
    
    def _normalize_audio(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """
        Normalize audio to target dB level.
        
        Args:
            audio: Audio to normalize
            target_db: Target dB level
        
        Returns:
            Normalized audio
        """
        import numpy as np
        
        # Calculate current dB level
        eps = 1e-10
        current_db = 20 * np.log10(np.max(np.abs(audio)) + eps)
        
        # Calculate gain
        gain = 10**((target_db - current_db) / 20)
        
        # Apply gain
        normalized_audio = audio * gain
        
        # Ensure no clipping
        if np.max(np.abs(normalized_audio)) > 1.0:
            normalized_audio = normalized_audio / np.max(np.abs(normalized_audio))
        
        return normalized_audio
    
    def _create_fallback_audio(self, output_path: str) -> None:
        """
        Create a fallback audio file in case of errors.
        
        Args:
            output_path: Path to save the audio
        """
        try:
            import numpy as np
            import soundfile as sf
            
            # Create a simple tone with beeps
            sr = 22050
            duration = 5.0
            t = np.linspace(0, duration, int(sr * duration), False)
            
            # Generate a signal with beeps
            signal = np.zeros_like(t)
            
            # Add beeps at 1-second intervals
            for i in range(5):
                beep_start = i * sr
                beep_end = beep_start + int(0.2 * sr)  # 200ms beep
                signal[beep_start:beep_end] = 0.5 * np.sin(2 * np.pi * 440 * t[beep_start:beep_end])
            
            # Save as WAV
            sf.write(output_path, signal, sr)
            
            logger.info(f"Created fallback audio at {output_path}")
        except Exception as e:
            logger.error(f"Failed to create fallback audio: {e}")
            
            # Create empty file as last resort
            with open(output_path, 'wb') as f:
                f.write(b'')