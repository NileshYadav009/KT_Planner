"""
Performance Optimization Module
- Faster transcription pipeline
- Parallel processing
- Intelligent caching
- Audio preprocessing optimization
"""

import os
import concurrent.futures
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta


class TranscriptionCache:
    """Smart caching for transcription results."""
    
    def __init__(self, cache_dir: str = ".cache/transcriptions"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "index.json"
        self.cache_ttl = timedelta(days=7)  # Cache valid for 7 days
        self._load_index()

    def _load_index(self):
        """Load cache index."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def _save_index(self):
        """Save cache index."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.index, f)

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate hash of audio file."""
        hash_obj = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def get(self, audio_file: str) -> Optional[Dict]:
        """Get cached transcription if available and valid."""
        try:
            file_hash = self._get_file_hash(audio_file)
            cache_key = file_hash
            
            if cache_key in self.index:
                entry = self.index[cache_key]
                cached_time = datetime.fromisoformat(entry['timestamp'])
                
                # Check TTL
                if datetime.utcnow() - cached_time < self.cache_ttl:
                    cache_file = self.cache_dir / entry['filename']
                    if cache_file.exists():
                        with open(cache_file) as f:
                            return json.load(f)
                else:
                    # Expired, remove
                    del self.index[cache_key]
                    self._save_index()
        except Exception as e:
            print(f"Cache get error: {e}")
        
        return None

    def set(self, audio_file: str, transcription: Dict) -> bool:
        """Cache transcription result."""
        try:
            file_hash = self._get_file_hash(audio_file)
            cache_key = file_hash
            cache_filename = f"{cache_key}.json"
            cache_path = self.cache_dir / cache_filename
            
            with open(cache_path, 'w') as f:
                json.dump(transcription, f)
            
            self.index[cache_key] = {
                'filename': cache_filename,
                'timestamp': datetime.utcnow().isoformat(),
                'source_file': audio_file
            }
            self._save_index()
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False


class ParallelProcessor:
    """Process audio in parallel segments for speed."""
    
    def __init__(self, max_workers: int = 4, segment_duration: int = 30):
        self.max_workers = max_workers
        self.segment_duration = segment_duration  # seconds

    def split_audio(self, audio_path: str) -> List[Tuple[int, str]]:
        """Split audio into segments for parallel processing."""
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_mp3(audio_path) if audio_path.endswith('.mp3') else AudioSegment.from_file(audio_path)
            
            segment_ms = self.segment_duration * 1000
            segments = []
            
            for i, start_ms in enumerate(range(0, len(audio), segment_ms)):
                end_ms = min(start_ms + segment_ms, len(audio))
                segment = audio[start_ms:end_ms]
                
                # Save temporary segment
                temp_path = f"/tmp/segment_{i}.mp3"
                segment.export(temp_path, format="mp3")
                segments.append((i, temp_path))
            
            return segments
        except ImportError:
            print("pydub not available, using sequential processing")
            return [(0, audio_path)]

    def transcribe_segments(self, model, segments: List[Tuple[int, str]]) -> List[Dict]:
        """Transcribe segments in parallel."""
        results = [None] * len(segments)
        
        def transcribe_one(idx, path):
            try:
                result = model.transcribe(path, language="en", verbose=False)
                return idx, result
            except Exception as e:
                print(f"Segment {idx} error: {e}")
                return idx, None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(transcribe_one, idx, path) for idx, path in segments]
            
            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                results[idx] = result
        
        # Clean up temp files
        for _, path in segments:
            if path != segments[0][1]:  # Don't delete original
                try:
                    os.remove(path)
                except:
                    pass
        
        return [r for r in results if r is not None]

    def merge_transcriptions(self, transcriptions: List[Dict]) -> Dict:
        """Merge parallel transcription results."""
        if not transcriptions:
            return {"text": "", "segments": []}
        
        # Sort by segment order (implicit in list order)
        merged_segments = []
        merged_text_parts = []
        time_offset = 0
        
        for trans in transcriptions:
            segments = trans.get('segments', [])
            for seg in segments:
                # Adjust timing
                adjusted_seg = seg.copy()
                adjusted_seg['start'] += time_offset
                adjusted_seg['end'] += time_offset
                merged_segments.append(adjusted_seg)
                merged_text_parts.append(seg.get('text', ''))
            
            if segments:
                time_offset = merged_segments[-1]['end']
        
        return {
            "text": " ".join(merged_text_parts),
            "segments": merged_segments,
            "language": "en"
        }


class AudioPreprocessor:
    """Fast audio preprocessing."""
    
    @staticmethod
    def quick_normalize(audio_path: str, output_path: str) -> bool:
        """Fast audio normalization."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(audio_path) if audio_path.endswith('.mp3') else AudioSegment.from_file(audio_path)
            
            # Normalize to -20dBFS
            from pydub.utils import mediainfo
            if audio.dBFS < -20:
                audio = audio.apply_gain(20 - abs(audio.dBFS))
            
            audio.export(output_path, format="mp3")
            return True
        except Exception as e:
            print(f"Normalization error: {e}")
            return False

    @staticmethod
    def detect_speech_regions(audio_path: str) -> List[Tuple[float, float]]:
        """Fast detection of speech regions (skip silence)."""
        try:
            import librosa
            import numpy as np
            
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Simple energy-based detection
            frame_length = 2048
            hop_length = 512
            
            S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Energy threshold
            energy = np.mean(S_db, axis=0)
            threshold = np.mean(energy) - 10
            
            # Find continuous speech regions
            speech_frames = energy > threshold
            speech_times = librosa.frames_to_time(np.where(speech_frames)[0], sr=sr, hop_length=hop_length)
            
            # Group into regions
            regions = []
            start = None
            for i, t in enumerate(speech_times):
                if start is None:
                    start = t
                elif t - speech_times[i-1] > 0.5:  # Gap > 500ms
                    regions.append((start, speech_times[i-1]))
                    start = t
            
            if start is not None:
                regions.append((start, speech_times[-1]))
            
            return regions
        except ImportError:
            return [(0, float('inf'))]  # Process entire audio

    @staticmethod
    def trim_silence(audio_path: str, output_path: str, threshold: int = -50) -> bool:
        """Trim leading/trailing silence."""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Simple amplitude-based trimming
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Find frames above threshold
            threshold_frame = S_db > threshold
            
            if threshold_frame.any():
                active_frames = np.where(threshold_frame.any(axis=0))[0]
                start_frame = active_frames[0]
                end_frame = active_frames[-1]
                
                start_time = librosa.frames_to_time(start_frame, sr=sr)
                end_time = librosa.frames_to_time(end_frame, sr=sr)
                
                trimmed = y[int(start_time*sr):int(end_time*sr)]
                
                import soundfile as sf
                sf.write(output_path, trimmed, sr)
                return True
            
            return False
        except ImportError:
            return False


class OptimizedTranscriptionPipeline:
    """Complete optimized transcription pipeline."""
    
    def __init__(self, model=None, enable_cache: bool = True, 
                 parallel: bool = True, preprocess: bool = True):
        self.model = model
        self.cache = TranscriptionCache() if enable_cache else None
        self.processor = ParallelProcessor() if parallel else None
        self.preprocessor = AudioPreprocessor() if preprocess else None
        self.stats = {}

    def transcribe_fast(self, audio_path: str) -> Dict:
        """Fast transcription with optimizations."""
        import time
        start_time = time.time()
        
        # Check cache
        if self.cache:
            cached = self.cache.get(audio_path)
            if cached:
                self.stats['cache_hit'] = True
                self.stats['transcription_time'] = time.time() - start_time
                return cached
        
        # Preprocess
        working_path = audio_path
        if self.preprocessor:
            # Try trimming silence
            trimmed_path = f"{audio_path}.trimmed.mp3"
            if self.preprocessor.trim_silence(audio_path, trimmed_path):
                working_path = trimmed_path
        
        # Transcribe
        if self.processor:
            # Parallel processing
            segments = self.processor.split_audio(working_path)
            if len(segments) > 1:
                transcriptions = self.processor.transcribe_segments(self.model, segments)
                result = self.processor.merge_transcriptions(transcriptions)
            else:
                result = self.model.transcribe(working_path, language="en", verbose=False)
        else:
            result = self.model.transcribe(working_path, language="en", verbose=False)
        
        # Cache result
        if self.cache:
            self.cache.set(audio_path, result)
        
        # Cleanup
        if working_path != audio_path:
            try:
                os.remove(working_path)
            except:
                pass
        
        self.stats['cache_hit'] = False
        self.stats['transcription_time'] = time.time() - start_time
        self.stats['method'] = 'parallel' if self.processor else 'sequential'
        
        return result

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return self.stats
