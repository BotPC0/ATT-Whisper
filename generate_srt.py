#!/usr/bin/env python3
"""
Whisper → SRT (and VTT) subtitle generator
------------------------------------------
Usage examples:

# Single file (Spanish, large-v3), auto-GPU:
python generate_srt.py "input/audio_or_video.mp3" -m large-v3 -l es

# Folder with many files, export SRT+VTT to ./subs:
python generate_srt.py "/path/to/folder" -m medium -l es -o "./subs" -f both

# Force CPU (if you have no CUDA):
python generate_srt.py "clip.mp4" -m base -l es --device cpu

Notes:
- Requires: ffmpeg, Python 3.9+, and the package "openai-whisper".
  Install:   pip install -U openai-whisper
  (GPU users should install PyTorch with CUDA per pytorch.org for best speed.)
- Output files keep the same base name as the input, with .srt/.vtt sidecars.
- Works with audio or video (ffmpeg demuxes the audio track).
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import whisper
from whisper.utils import get_writer

def list_media_files(path: Path):
    exts = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma",
            ".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi"}
    if path.is_file():
        return [path]
    files = []
    for p in path.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="Generate subtitles (.srt/.vtt) from audio/video using Whisper.")
    parser.add_argument("input", type=str, help="Input media file OR folder containing media files.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Directory to save subtitle files (defaults to input's folder).")
    parser.add_argument("-m", "--model", type=str, default="large-v3",
                        help="Whisper model size (tiny, base, small, medium, large-v2, large-v3). Default: large-v3")
    parser.add_argument("-l", "--language", type=str, default="es", help="Language code (e.g., es, en, fr). Default: es")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="transcribe (same language) or translate to English. Default: transcribe")
    parser.add_argument("-f", "--format", type=str, choices=["srt", "vtt", "both"], default="srt",
                        help="Subtitle format to export. Default: srt")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto",
                        help="Computation device. Default: auto")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Lower = more deterministic.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for decoding. Default: 5")
    parser.add_argument("--verbose", action="store_true", help="Print segment-by-segment output in console.")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Custom directory to cache/download Whisper model weights (defaults to ~/.cache/whisper).")
    parser.add_argument("--max-words", type=int, default=None,
                        help="Maximum number of words per subtitle block (TikTok style). If used, generates *_chunked.srt/.vtt files.")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Maximum number of characters per subtitle block (applies after word limit if both are set).")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    fp16 = device == "cuda"

    in_path = Path(args.input).expanduser().resolve()
    files = list_media_files(in_path)
    if not files:
        print(f"[ERROR] No media files found at: {in_path}")
        sys.exit(1)

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        # If single file, keep in its directory; if folder, create 'subs' inside it
        if in_path.is_file():
            out_dir = in_path.parent
        else:
            out_dir = in_path / "subs"
            out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}  |  FP16: {fp16}")
    # Optional custom model directory
    download_root = None
    if args.model_dir:
        download_root = Path(args.model_dir).expanduser().resolve()
        download_root.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Using custom model dir: {download_root}")

    print(f"[INFO] Loading Whisper model: {args.model} …")
    model = whisper.load_model(args.model, device=device, download_root=str(download_root) if download_root else None)

    # Writers
    writers = []
    if args.format in ("srt", "both"):
        writers.append(("srt", get_writer("srt", str(out_dir))))
    if args.format in ("vtt", "both"):
        writers.append(("vtt", get_writer("vtt", str(out_dir))))

    common_options = dict(task=args.task, language=args.language, temperature=args.temperature, beam_size=args.beam_size, fp16=fp16)

    def split_into_chunks(segments, max_words: int | None, max_chars: int | None):
        """Splits the transcription (at word level if available) into small blocks.
        Each block keeps the timestamp from the first word to the last.
        If word-level timestamps are not available, it falls back to approximating by splitting the segment text.
        """
        if not max_words and not max_chars:
            return None  # No chunking requested

        # Whisper >= 202311 uses result["segments"][i]["words"] when word_timestamps enabled.
        # To ensure word-level timestamps, we will request word_timestamps=True in transcribe if chunking is requested.
        chunk_list = []
        idx = 1
        for seg in segments:
            words = seg.get("words")
            if not words:
                # Fallback: split plain text
                plain_words = seg["text"].strip().split()
                # Approximate equally spaced times
                start = seg["start"]
                end = seg["end"]
                dur = max(end - start, 0.0001)
                per_word = dur / max(len(plain_words), 1)
                words = []
                t_cursor = start
                for w in plain_words:
                    w_end = t_cursor + per_word
                    words.append({"word": w, "start": t_cursor, "end": w_end})
                    t_cursor = w_end

            # Ahora agrupamos
            current_words = []
            current_len = 0
            for w in words:
                w_text = w.get("word", "").strip()
                if not w_text:
                    continue
                tentative_words = current_words + [w]
                word_count = len(tentative_words)
                char_count = sum(len(x.get("word","")) for x in tentative_words) + (word_count - 1)
                if (max_words and word_count > max_words) or (max_chars and char_count > max_chars):
                    # cerrar bloque actual
                    if current_words:
                        block_text = " ".join(x["word"].strip() for x in current_words).strip()
                        chunk_list.append({
                            "index": idx,
                            "start": current_words[0]["start"],
                            "end": current_words[-1]["end"],
                            "text": block_text
                        })
                        idx += 1
                    current_words = [w]
                else:
                    current_words = tentative_words
            # flush final
            if current_words:
                block_text = " ".join(x["word"].strip() for x in current_words).strip()
                chunk_list.append({
                    "index": idx,
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"],
                    "text": block_text
                })
                idx += 1
        return chunk_list

    def format_timestamp(ts: float):
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts - int(ts)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def write_chunked_srt(chunks, out_path: Path):
        lines = []
        for c in chunks:
            lines.append(str(c["index"]))
            lines.append(f"{format_timestamp(c['start'])} --> {format_timestamp(c['end'])}")
            lines.append(c["text"])
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")

    def write_chunked_vtt(chunks, out_path: Path):
        lines = ["WEBVTT", ""]
        for c in chunks:
            # VTT timestamp uses . instead of , for ms
            def vtt_time(ts: float):
                h = int(ts // 3600)
                m = int((ts % 3600) // 60)
                s = int(ts % 60)
                ms = int((ts - int(ts)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
            lines.append(f"{vtt_time(c['start'])} --> {vtt_time(c['end'])}")
            lines.append(c["text"])
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")

    need_chunking = args.max_words or args.max_chars

    for i, media in enumerate(files, 1):
        print(f"\n=== [{i}/{len(files)}] {media.name} ===")
        try:
            # Transcribe
            transcribe_opts = dict(common_options)
            if need_chunking:
                transcribe_opts["word_timestamps"] = True
            result = model.transcribe(str(media), **transcribe_opts, verbose=args.verbose)

            # Save subtitles
            for fmt, w in writers:
                # The writer will create <basename>.<fmt> in out_dir
                w(result, str(media))
                out_file = out_dir / f"{media.stem}.{fmt}"
                if out_file.exists():
                    print(f"[OK] Saved: {out_file}")
                else:
                    print(f"[WARN] Expected output not found: {out_file}")

            # TikTok style chunked captions
            if need_chunking:
                chunks = split_into_chunks(result.get("segments", []), args.max_words, args.max_chars)
                if chunks:
                    if args.format in ("srt", "both"):
                        chunked_srt = out_dir / f"{media.stem}_chunked.srt"
                        write_chunked_srt(chunks, chunked_srt)
                        print(f"[OK] Chunked SRT: {chunked_srt}")
                    if args.format in ("vtt", "both"):
                        chunked_vtt = out_dir / f"{media.stem}_chunked.vtt"
                        write_chunked_vtt(chunks, chunked_vtt)
                        print(f"[OK] Chunked VTT: {chunked_vtt}")
        except Exception as e:
            print(f"[ERROR] Failed on {media.name}: {e}")

    print("\nDone. Import the .srt/.vtt in Adobe Premiere Pro (File → Import) and drop it into your sequence or your preferred software.")

if __name__ == "__main__":
    main()
