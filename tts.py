# tts.py
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/SSAFY/Desktop/work/onion_PJT/S12P31S307/src/AI/TTS_Similarity/autodubbing_2ver/thewater-455706-f98cc73703ca.json"

class TextToSpeech:
    """Class for synthesizing speech using TTS engines."""
    
    def __init__(
        self, 
        lang: str = "en", 
        voice_id: Optional[str] = None,
        engine: str = "neural",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        voice_style: str = "neutral"
    ):
        """
        Initialize the TTS engine.
        
        Args:
            lang: Language code
            voice_id: Voice ID to use (if None, a default is selected)
            engine: TTS engine to use ('neural', 'amazon', 'google', etc.)
            speaking_rate: Speaking rate multiplier (1.0 = normal)
            pitch: Pitch adjustment (-10.0 to 10.0)
            volume: Volume (0.0 to 2.0)
            voice_style: Voice style ('neutral', 'conversational', 'formal')
        """
        self.lang = lang
        self.voice_id = voice_id
        self.engine = engine
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self.volume = volume
        self.voice_style = voice_style
        
        # Map language codes to voice IDs for different engines
        self.voice_map = {
            "neural": {
                "en": "en-US-Neural2-F",
                "fr": "fr-FR-Neural2-F",
                "de": "de-DE-Neural2-F",
                "es": "es-ES-Neural2-F",
                "it": "it-IT-Neural2-F",
                "ko": "ko-KR-Neural2-F"
            },
            "amazon": {
                "en": "Joanna",
                "fr": "Léa",
                "de": "Vicki",
                "es": "Lucia",
                "it": "Carla",
                "ko": "Seoyeon"
            },
            "google": {
                "en": "en-US-Wavenet-F",
                "fr": "fr-FR-Wavenet-C",
                "de": "de-DE-Wavenet-F",
                "es": "es-ES-Wavenet-C",
                "it": "it-IT-Wavenet-B",
                "ko": "ko-KR-Wavenet-B"
            }
        }
        
        # Load TTS engine
        self._init_tts_engine()
        
        # Select voice
        if voice_id is None:
            self.voice_id = self.voice_map.get(self.engine, {}).get(self.lang, None)
        
        logger.info(f"Initialized {self.engine} TTS for language {self.lang} with voice {self.voice_id}")
        logger.info(f"TTS parameters: speaking_rate={speaking_rate}, pitch={pitch}, volume={volume}, style={voice_style}")
    
    # [Rest of the class methods remain the same]
    
    def _synthesize_neural(
        self, 
        text: str, 
        output_path: str, 
        target_duration: Optional[float] = None
    ) -> float:
        """
        Synthesize speech using a neural TTS engine.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the output audio
            target_duration: Target duration in seconds
        
        Returns:
            Actual duration of the synthesized audio
        """
        try:
            # Generate speech with specified parameters
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker=self.voice_id,
                language=self.lang,
                # Apply style and other parameters if supported by the engine
                speed=self.speaking_rate,
                # Additional parameters might be supported by specific TTS engines
            )
            
            # Get actual duration
            import librosa
            duration = librosa.get_duration(path=output_path)
            
            # Adjust duration if needed
            if target_duration is not None and abs(duration - target_duration) > 0.1:
                self._adjust_audio_duration(output_path, target_duration)
                duration = target_duration
            
            return duration
        except Exception as e:
            logger.error(f"Error in neural TTS: {e}")
            return self._synthesize_mock(text, output_path, target_duration)
    
    def _init_tts_engine(self):
        """Initialize the TTS engine based on the selected type."""
        try:
            if self.engine == "neural":
                self._init_neural_tts()
            elif self.engine == "amazon":
                self._init_amazon_tts()
            elif self.engine == "google":
                self._init_google_tts()
            else:
                logger.warning(f"Unknown TTS engine: {self.engine}, falling back to neural TTS")
                self.engine = "neural"
                self._init_neural_tts()
        except Exception as e:
            logger.error(f"Failed to initialize {self.engine} TTS: {e}")
            logger.info("Falling back to mock TTS implementation")
            self.use_mock = True
    
    def _init_neural_tts(self):
        """Initialize a local neural TTS engine (e.g., using TTS or ESPnet)."""
        try:
            # Try to import TTS library
            import torch
            from TTS.api import TTS
            
            # Initialize TTS
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
            self.use_mock = False
        except ImportError:
            logger.warning("TTS library not available, using mock implementation")
            self.use_mock = True
    
    def _init_amazon_tts(self):
        """Initialize Amazon Polly TTS."""
        try:
            import boto3
            
            # Initialize Polly client
            self.polly_client = boto3.client('polly')
            self.use_mock = False
        except ImportError:
            logger.warning("boto3 not available, using mock implementation")
            self.use_mock = True
    
    def _init_google_tts(self):
        """Initialize Google TTS."""
        try:
            from google.cloud import texttospeech
            
            # 클라이언트 초기화
            self.google_client = texttospeech.TextToSpeechClient()
            self.use_mock = False
            logger.info("Google Cloud TTS initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS: {e}")
            logger.warning("Google Cloud TTS not available, using mock implementation")
            self.use_mock = True
    
    def synthesize(
        self, 
        sentences: List[str], 
        durations: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Synthesize speech for the given sentences.
        
        Args:
            sentences: List of sentences to synthesize
            durations: Optional list of target durations for each sentence
        
        Returns:
            List of dictionaries with audio path and metadata
        """
        results = []
        
        # Create temporary directory for audio files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for i, sentence in enumerate(sentences):
                # Get target duration if provided
                target_duration = None
                if durations is not None and i < len(durations):
                    target_duration = durations[i]
                
                # Generate speech
                audio_file = temp_path / f"speech_{i}.wav"
                
                if self.use_mock:
                    # Use mock implementation
                    duration = self._synthesize_mock(sentence, str(audio_file), target_duration)
                else:
                    # Use real TTS engine
                    if self.engine == "neural":
                        duration = self._synthesize_neural(sentence, str(audio_file), target_duration)
                    elif self.engine == "amazon":
                        duration = self._synthesize_amazon(sentence, str(audio_file), target_duration)
                    elif self.engine == "google":
                        duration = self._synthesize_google(sentence, str(audio_file), target_duration)
                    else:
                        duration = self._synthesize_mock(sentence, str(audio_file), target_duration)
                
                # Create a more permanent copy of the audio file
                output_dir = Path("output/audio")
                output_dir.mkdir(exist_ok=True, parents=True)
                
                output_file = output_dir / f"speech_{i}.wav"
                
                # Copy the file to the output directory
                import shutil
                shutil.copy(audio_file, output_file)
                
                results.append({
                    "text": sentence,
                    "audio_path": str(output_file),
                    "duration": duration,
                    "target_duration": target_duration
                })
                
                logger.info(f"Synthesized speech for sentence {i+1}/{len(sentences)}: {sentence[:30]}...")
        
        return results
    
    def _synthesize_neural(
        self, 
        text: str, 
        output_path: str, 
        target_duration: Optional[float] = None
    ) -> float:
        """
        Synthesize speech using a neural TTS engine.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the output audio
            target_duration: Target duration in seconds
        
        Returns:
            Actual duration of the synthesized audio
        """
        try:
            # Generate speech
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker=self.voice_id,
                language=self.lang
            )
            
            # Get actual duration
            import librosa
            duration = librosa.get_duration(path=output_path)
            
            # Adjust duration if needed
            if target_duration is not None and abs(duration - target_duration) > 0.1:
                self._adjust_audio_duration(output_path, target_duration)
                duration = target_duration
            
            return duration
        except Exception as e:
            logger.error(f"Error in neural TTS: {e}")
            return self._synthesize_mock(text, output_path, target_duration)
    
    def _synthesize_amazon(
        self, 
        text: str, 
        output_path: str, 
        target_duration: Optional[float] = None
    ) -> float:
        """
        Synthesize speech using Amazon Polly.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the output audio
            target_duration: Target duration in seconds
        
        Returns:
            Actual duration of the synthesized audio
        """
        try:
            # Generate speech
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat='pcm',
                VoiceId=self.voice_id,
                Engine='neural'
            )
            
            # Save audio
            with open(output_path, 'wb') as file:
                file.write(response['AudioStream'].read())
            
            # Get actual duration
            import librosa
            duration = librosa.get_duration(path=output_path)
            
            # Adjust duration if needed
            if target_duration is not None and abs(duration - target_duration) > 0.1:
                self._adjust_audio_duration(output_path, target_duration)
                duration = target_duration
            
            return duration
        except Exception as e:
            logger.error(f"Error in Amazon Polly TTS: {e}")
            return self._synthesize_mock(text, output_path, target_duration)
    
    def _synthesize_google(
        self, 
        text: str, 
        output_path: str, 
        target_duration: Optional[float] = None
    ) -> float:
        """
        Synthesize speech using Google TTS.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the output audio
            target_duration: Target duration in seconds
        
        Returns:
            Actual duration of the synthesized audio
        """
        try:
            from google.cloud import texttospeech
            
            # Set up request
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build voice parameters
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.lang,
                name=self.voice_id
            )
            
            # Select audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )
            
            # Generate speech
            response = self.google_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Save audio
            with open(output_path, 'wb') as file:
                file.write(response.audio_content)
            
            # Get actual duration
            import librosa
            duration = librosa.get_duration(path=output_path)
            
            # Adjust duration if needed
            if target_duration is not None and abs(duration - target_duration) > 0.1:
                self._adjust_audio_duration(output_path, target_duration)
                duration = target_duration
            
            return duration
        except Exception as e:
            logger.error(f"Error in Google TTS: {e}")
            return self._synthesize_mock(text, output_path, target_duration)
    
    def _synthesize_mock(
        self, 
        text: str, 
        output_path: str, 
        target_duration: Optional[float] = None
    ) -> float:
        """
        Create a mock audio file for testing.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the output audio
            target_duration: Target duration in seconds
        
        Returns:
            Actual duration of the synthesized audio
        """
        try:
            import numpy as np
            from scipy.io import wavfile
            
            # Estimate natural duration
            words = len(text.split())
            natural_duration = max(1.0, words * 0.3)  # ~0.3 seconds per word
            
            # Use target duration if provided
            duration = target_duration if target_duration is not None else natural_duration
            
            # Generate a simple sine wave
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Generate a note that changes pitch based on text length
            frequency = 440 + (len(text) % 10) * 20
            audio = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            # Add some random noise
            audio += 0.01 * np.random.normal(0, 1, len(t))
            
            # Ensure the audio is within range [-1, 1]
            audio = np.clip(audio, -1, 1)
            
            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Save as WAV
            wavfile.write(output_path, sample_rate, audio_int16)
            
            return duration
        except Exception as e:
            logger.error(f"Error in mock TTS: {e}")
            
            # Create an empty file as fallback
            with open(output_path, 'wb') as file:
                file.write(b'')
            
            return 1.0  # Default duration
    
    def _adjust_audio_duration(self, audio_path: str, target_duration: float) -> None:
        """
        Adjust the duration of an audio file.
        
        Args:
            audio_path: Path to the audio file
            target_duration: Target duration in seconds
        """
        try:
            import librosa
            import soundfile as sf
            import pyrubberband as pyrb
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Get current duration
            current_duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate stretch factor
            stretch_factor = target_duration / current_duration
            
            # Only adjust if the difference is significant
            if abs(stretch_factor - 1.0) > 0.05:
                # Apply time stretching
                y_stretched = pyrb.time_stretch(y, sr, stretch_factor)
                
                # Save the stretched audio
                sf.write(audio_path, y_stretched, sr)
                
                logger.info(f"Adjusted audio duration from {current_duration:.2f}s to {target_duration:.2f}s")
        except Exception as e:
            logger.error(f"Error adjusting audio duration: {e}")