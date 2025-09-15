# ATT-Whisper Subtitle Generator

Generate SRT (and VTT) subtitles from audio or video files using OpenAI Whisper.

## Features
- Single file or whole folder processing (recursive media discovery)
- Choose Whisper model size (`tiny` .. `large-v3`)
- Spanish by default (override with `-l <lang>`)
- Transcribe or translate (`--task translate` → English)
- Export `.srt`, `.vtt`, or both
- GPU auto-detection (CUDA → FP16 for speed) or force CPU
- Custom model weights directory via `--model-dir`
 - Generación de subtítulos en bloques pequeños tipo TikTok / Shorts con `--max-words` y `--max-chars`

## Requirements
- Windows (tested) / Linux / macOS
- Python 3.9+ (you are using 3.12)
- FFmpeg installed and on PATH
- Python packages: `openai-whisper`, `torch` (+ CUDA build if using GPU)

Install packages (already done in your environment):
```powershell
py -3.12 -m pip install -U pip openai-whisper
```
If you need CUDA-enabled Torch for a new setup, follow instructions at: https://pytorch.org/

## Basic Usage
Single file:
```powershell
py -3.12 generate_srt.py "input.mp3" -m small -l es
```
Folder (creates `subs/` inside folder):
```powershell
py -3.12 generate_srt.py "D:\Media\Interviews" -m medium -l es -f both
```
Force CPU:
```powershell
py -3.12 generate_srt.py "clip.mp4" -m base --device cpu
```
Translate to English:
```powershell
py -3.12 generate_srt.py "entrevista.wav" -m medium --task translate -l es
```

## TikTok / Shorts Chunked Captions
Para generar bloques muy cortos (por ejemplo 3–5 palabras) que se vean mejor en vertical:

```powershell

# Maximum 4 words per block
py -3.12 generate_srt.py "video.mp4" -m small -l es --max-words 4

# Maximum 5 words or 28 characters (whichever comes first)
py -3.12 generate_srt.py "video.mp4" -m medium -l es --max-words 5 --max-chars 28

# Character Limit Only (No Word Limit)
py -3.12 generate_srt.py "video.mp4" -m base -l es --max-chars 32
```

Archivos producidos (además de los normales):
- `nombre_chunked.srt`
- `nombre_chunked.vtt` (si usas `-f vtt` o `-f both`)

Cómo funciona el corte:
1. Activa `word_timestamps` para obtener marcas de tiempo por palabra (si el backend lo soporta).
2. Acumula palabras hasta que se excedería `--max-words` o `--max-chars`.
3. Cierra el bloque y abre uno nuevo.
4. Si el modelo no devuelve palabras individuales, hace un reparto aproximado dentro del segmento.

Sugerencias:
- 3–5 palabras suelen ser legibles para ritmos rápidos.
- Combina un límite de palabras y caracteres para evitar líneas muy largas.
- Revisa el resultado y ajusta valores según el estilo del video.

## Example (Your Workflow)
Exact command you ran:
```powershell
py -3.12 generate_srt.py "C:\PATH" -m medium -l es -f srt --device auto --model-dir ".\models" --max-words 4 --max-chars 10
```
What happens step-by-step:
1. Script resolves device: selects `cuda` if available else `cpu`.
2. Creates/uses custom model cache folder: `.\models` (so weights like `medium.pt` live in the project).
3. Loads the Whisper `medium` model (downloads once if missing).
4. Processes the WAV file (or all supported media if a folder given).
5. Generates segments and writes `tomate_pesto_voz.prproj_mixdown_Samson.srt` next to the input (or `subs/` for folder mode).
6. You can import the SRT into Adobe Premiere Pro.

## Supported Media Extensions
Audio: mp3, wav, m4a, aac, flac, ogg, wma
Video: mp4, mov, mkv, m4v, webm, avi

## Output Location Rules
- Single file input → subtitle saved alongside the media.
- Folder input → a `subs/` directory is created inside that folder unless `-o` is provided.
- Custom output directory: use `-o path/to/output`.

## Custom Model Directory
Use `--model-dir` to store/download model weights locally (helpful for portability or offline use):
```powershell
py -3.12 generate_srt.py input.wav -m medium --model-dir .\models
```
Pre-seed by moving an existing weight file:
```powershell
Move-Item $env:USERPROFILE\.cache\whisper\medium.pt .\models\medium.pt
```

## Choosing a Model
| Model | Size | Notes |
|-------|------|-------|
| tiny | ~75 MB | Fastest, lowest accuracy |
| base | ~142 MB | Lightweight |
| small | ~466 MB | Good balance |
| medium | ~1.5 GB | Higher accuracy (your current choice) |
| large-v2 | ~3.1 GB | Older large option |
| large-v3 | ~3.2 GB | Latest high accuracy |

If VRAM is limited or you only need a quick draft, try `small`.

## Language and Translation
- `-l es` sets source language (Speeds up + improves accuracy vs auto-detect).
- `--task translate` outputs English text from the source language.

## Performance Tips
- Use GPU (`cuda`) for large models. Automatically chosen when available.
- FP16 is enabled on CUDA to reduce memory and improve speed.
- For long batches of files, consider smaller models to reduce load times.

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| FFmpeg not found | Not installed / not on PATH | Install from https://www.gyan.dev/ffmpeg/builds/ and add `ffmpeg\bin` to PATH |
| CUDA not used | Driver / toolkit mismatch | Update NVIDIA driver; ensure you installed correct Torch CUDA build |
| Out of VRAM | Model too large | Try `small` or `medium` instead of `large-*` |
| Slow CPU run | Using big model on CPU | Switch to smaller model |
| Wrong language output | Language mis-specified | Adjust `-l` or remove to auto-detect |

## Script Arguments Summary
```text
positional:
  input                 File or folder

options:
  -o, --output_dir      Output directory
  -m, --model           Model size (default: large-v3)
  -l, --language        Language code (default: es)
      --task            transcribe | translate (default: transcribe)
  -f, --format          srt | vtt | both (default: srt)
      --device          auto | cuda | cpu (default: auto)
      --temperature     Sampling temperature (default: 0.0)
      --beam_size       Beam size (default: 5)
      --verbose         Print segments to console
      --model-dir       Custom model download/cache directory
  --max-words       Máximo de palabras por bloque (genera *_chunked.srt/.vtt)
  --max-chars       Máximo de caracteres por bloque (aplicado después del límite de palabras)
```

## Roadmap Ideas
- Add progress bar per file (currently only the model download + decode progress are shown)
- Optional JSON export of segments
- Auto language detection option

## License / Usage
Use responsibly. Whisper models are released under the MIT license (see upstream repo). Check media rights before transcribing.

---
Questions or want enhancements (batch script, preload, JSON output)? Open an issue or ask.
