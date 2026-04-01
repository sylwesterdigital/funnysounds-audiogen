**prompt → local AI model writes WAV → ffmpeg mixes WAV into the video**

For a **fully local** setup, the cleanest option is **Meta AudioCraft / AudioGen**. It is explicitly a **text-to-sound** model, AudioCraft recommends having `ffmpeg` installed, and the documented pretrained model is `facebook/audiogen-medium`. The main caveat is that AudioCraft’s **code is MIT**, but the **model weights are CC-BY-NC 4.0**, and Meta says the medium model needs a **GPU with at least 16 GB VRAM** for inference. ([GitHub][1])

If you need something friendlier for **commercial use**, use **Stable Audio Open 1.0** or Stability’s **Stable Audio 2.5** instead. Stability says Stable Audio Open 1.0 can generate **up to 47 seconds** of **44.1 kHz stereo** audio from text and can be used through either **`stable-audio-tools`** or **`diffusers`**; Stability’s API docs say Stable Audio 2.5 supports **text-to-audio, audio-to-audio, and audio inpainting**. ([Hugging Face][2])

Here is the most practical **local AudioGen + ffmpeg** version.

### 1) Install AudioGen locally

```bash
python -m pip install 'torch==2.1.0'
python -m pip install setuptools wheel
python -m pip install -U audiocraft
```

AudioCraft’s own install notes also recommend having `ffmpeg` installed. ([GitHub][1])

### 2) Create a tiny generator script

Save as `gen_sfx.py`:

```python
import sys
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

prompt = sys.argv[1] if len(sys.argv) > 1 else "cartoon rubber squeak, wet splat, absurd comedy impact"
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
out = sys.argv[3] if len(sys.argv) > 3 else "funny_sfx"

model = AudioGen.get_pretrained("facebook/audiogen-medium")
model.set_generation_params(duration=duration)

wav = model.generate([prompt])

audio_write(
    out,
    wav[0].cpu(),
    model.sample_rate,
    strategy="loudness",
    loudness_compressor=True
)
```

That is the same AudioGen API shape Meta documents: `AudioGen.get_pretrained(...)`, `set_generation_params(duration=...)`, `generate(...)`, and `audio_write(...)`. ([GitHub][3])

### 3) Generate a funny sound

```bash
python gen_sfx.py "rubber chicken squeak, slimy splat, tiny alien panic, cartoon fail sting" 2.5 sfx1
```

That writes `sfx1.wav`.

### 4) Mix it into your video with ffmpeg

Put the sound at **3.2 seconds**:

```bash
ffmpeg -y \
  -i input.mp4 \
  -i sfx1.wav \
  -filter_complex "[1:a]adelay=3200|3200,volume=0.9[sfx];[0:a][sfx]amix=inputs=2:duration=first:dropout_transition=0[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -shortest output.mp4
```

### 5) One-command workflow

Save as `add_funny_sfx.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:-input.mp4}"
PROMPT="${2:-cartoon rubber squeak, wet splat, ridiculous comedy hit}"
TIME_MS="${3:-3200}"

python gen_sfx.py "$PROMPT" 2.5 tmp_funny
ffmpeg -y \
  -i "$VIDEO" \
  -i tmp_funny.wav \
  -filter_complex "[1:a]adelay=${TIME_MS}|${TIME_MS},volume=0.9[sfx];[0:a][sfx]amix=inputs=2:duration=first:dropout_transition=0[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -shortest "${VIDEO%.*}_funny.mp4"
```

Run:

```bash
bash add_funny_sfx.sh input.mp4 "evil sausage attack, squishy footsteps, mustard explosion, absurd comedy sting" 4100
```

For **best prompts**, keep them short and physical:

* `cartoon stumble, rubber squeak, then silly boing`
* `wet splat in mud, plastic wobble, comic impact`
* `tiny UFO panic, toy synth chirp, ridiculous sci-fi blip`
* `sausage monster attack, squishy footsteps, ketchup burst`

If you want **commercial-safe** instead of AudioGen, switch the generator step to **Stable Audio Open** or **Stable Audio 2.5**; AudioGen is the easiest local hack, but its released weights are not the safest choice for commercial shipping. ([GitHub][4])

Paste your current ffmpeg command and it can be merged into a single script.

[1]: https://github.com/facebookresearch/audiocraft "GitHub - facebookresearch/audiocraft: Audiocraft is a library for audio processing and generation with deep learning. It features the state-of-the-art EnCodec audio compressor / tokenizer, along with MusicGen, a simple and controllable music generation LM with textual and melodic conditioning. · GitHub"
[2]: https://huggingface.co/stabilityai/stable-audio-open-1.0 "stabilityai/stable-audio-open-1.0 · Hugging Face"
[3]: https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md "audiocraft/docs/AUDIOGEN.md at main · facebookresearch/audiocraft · GitHub"
[4]: https://github.com/facebookresearch/audiocraft/blob/main/model_cards/AUDIOGEN_MODEL_CARD.md "audiocraft/model_cards/AUDIOGEN_MODEL_CARD.md at main · facebookresearch/audiocraft · GitHub"
