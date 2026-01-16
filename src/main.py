import argparse
import hashlib
import logging
import os
import pickle
import random
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import Voice, play, save
from elevenlabs.client import ElevenLabs

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def main():
    load_dotenv(Path(__file__).parent.parent / ".env")

    CACHE_DIR = (
        Path(os.getenv("XDG_CACHE_HOME", Path(os.environ["HOME"]) / ".cache"))
        / "elevenlabs"
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_VOICE_NAME = os.getenv("DEFAULT_VOICE_NAME", "Alice")

    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    voices = get_voices(client, CACHE_DIR)
    logger.debug(f"Voices: {[v.name for v in voices]}")

    args = parse_args(["Any", "Male", "Female", "All"] + [str(v.name) for v in voices])
    if args.debug:
        logger.setLevel(logging.DEBUG)

    text = " ".join(args.text)
    model_id = get_latest_model(client, CACHE_DIR)

    if args.voice == "All":
        success = False
        for voice in voices:
            logger.debug(voice.name)
            success = say(client, text, voice, model_id, CACHE_DIR)

        return 0 if success else 1

    voice_choices = None
    if args.voice == "Any":
        voice_choices = [v.name for v in voices]

    if args.voice in {"Male", "Female"}:
        voice_choices = [
            v
            for v in voices
            if v.labels and v.labels.get("gender") == args.voice.lower()
        ]

    if voice_choices:
        voice = random.choice(voice_choices)
    else:
        voice_name = args.voice or DEFAULT_VOICE_NAME
        voice = next(v for v in voices if v.name and voice_name in v.name)

    assert type(voice) is Voice

    success = say(client, text, voice, model_id, CACHE_DIR)

    return 0 if success else 1


def get_voices(client: ElevenLabs, cache_dir: Path):
    voices_path = Path(os.path.join(cache_dir, "voices.pickle"))

    if voices_path.is_file() and False:
        logger.debug("Loading voices from cache")
        with open(voices_path, "rb") as fp:
            voices = pickle.load(fp)
    else:
        logger.debug("Fetching voices from API")
        voices = client.voices.search().voices

        with open(voices_path, "wb") as fp:
            pickle.dump(voices, fp)

    return voices


def parse_args(voice_names: list[str]):
    parser = argparse.ArgumentParser(
        "say", description="Convert text to audible speech"
    )
    parser.add_argument("text", nargs="+")
    parser.add_argument("-v", "--voice", type=str, choices=voice_names)
    parser.add_argument("-d", "--debug", action="store_true")

    return parser.parse_args()


def get_latest_model(client: ElevenLabs, cache_dir: Path):
    models_path = Path(os.path.join(cache_dir, "models.pickle"))

    if models_path.is_file():
        logger.debug("Loading models from cache")
        with open(models_path, "rb") as fp:
            models = pickle.load(fp)
    else:
        logger.debug("Fetching models from API")
        try:
            models = client.models.list()
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            models = []
        with open(models_path, "wb") as fp:
            pickle.dump(models, fp)

    logger.info(f"Found {len(models)} models")
    logger.debug({m.model_id for m in models})

    flash_models = [m for m in models if "flash" in m.model_id]
    logger.info(f"Found {len(flash_models)} flash models")
    logger.debug({m.model_id for m in flash_models})

    if flash_models:
        return flash_models[0].model_id

    if models:
        return models[0].model_id

    return "eleven_flash_v2_5"


def say(client: ElevenLabs, text: str, voice: Voice, model_id: str, cache_dir: Path):
    logger.debug(f"Picking voice: {voice.name}")

    key = f"{voice.name}:{text}"
    hash = hashlib.sha256(key.encode()).hexdigest()

    filepath = cache_dir / "audio" / hash
    filepath = filepath.with_suffix(".mp3")

    if not filepath.is_file():
        logger.debug("Generating and playing audio")

        try:
            # audio = client.generate(text=text, voice=voice, model=model_id)
            audio = client.text_to_speech.convert(
                text=text,
                voice_id=voice.voice_id,
                model_id=model_id,
            )
            save(audio, str(filepath))

        except Exception as e:
            logger.error("Failed to generate audio")
            logger.exception(e)

            return False
    else:
        logger.debug(f"Playing audio from cache: {filepath}")

    try:
        with open(filepath, "rb") as fp:
            play(fp)

    except Exception as e:
        logger.error("Failed to play audio")
        logger.exception(e)
        logger.debug("Removing cache audio file")
        filepath.unlink()

        return False

    return True


if __name__ == "__main__":
    exit(main())
