import csv
import random
import requests
import os

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

LANG_FILES = {
    "hi": [
        "hi-1.txt",
        "hi-2.txt",
        "hi-3.txt",
    ],
    "bn": ["bn.txt"],
    "gu": ["gu.txt"],
    "ta": ["ta.txt"],
    "te": ["te.txt"],
}

N_PER_LANG = 4048
SEED = 42

OUTPUT = "data/indic_mrl/indic_mrl_train.csv"

BASE_URL = (
    "https://huggingface.co/datasets/"
    "ai4bharat/IndicCorpV2/resolve/main/data/"
)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)


# ---------------------------------------------------------
# Download only a bounded prefix of a remote file
# ---------------------------------------------------------

def download_prefix(filename, max_bytes=200 * 1024 * 1024):

    url = BASE_URL + filename

    print(f"\nReading: {filename}")
    print(url)

    headers = {
        "Range": f"bytes=0-{max_bytes - 1}"
    }

    response = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    chunks = []

    total = 0

    for chunk in response.iter_content(
        chunk_size=1024 * 1024
    ):

        if not chunk:
            continue

        chunks.append(chunk)
        total += len(chunk)

        if total >= max_bytes:
            break

    data = b"".join(chunks)

    print(
        f"Downloaded approximately "
        f"{len(data) / 1024 / 1024:.1f} MB"
    )

    return data.decode(
        "utf-8",
        errors="ignore",
    )


# ---------------------------------------------------------
# Extract sentences
# ---------------------------------------------------------

def get_sentences_from_prefix(filename):

    text = download_prefix(filename)

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if len(line) < 10:
            continue

        lines.append(line)

    print(
        f"Usable sentences from {filename}: "
        f"{len(lines):,}"
    )

    return lines


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    rng = random.Random(SEED)

    all_rows = []

    for lang, files in LANG_FILES.items():

        print()
        print("=" * 60)
        print(f"LANGUAGE: {lang}")
        print("=" * 60)

        candidates = []

        for filename in files:

            sentences = get_sentences_from_prefix(
                filename
            )

            candidates.extend(sentences)

        print(
            f"{lang}: total candidates = "
            f"{len(candidates):,}"
        )

        if len(candidates) < N_PER_LANG:

            raise RuntimeError(
                f"Not enough sentences for {lang}. "
                f"Found {len(candidates)}, "
                f"need {N_PER_LANG}."
            )

        # Deterministic random sample
        selected = rng.sample(
            candidates,
            N_PER_LANG,
        )

        print(
            f"{lang}: selected = "
            f"{len(selected):,}"
        )

        for sentence in selected:
            all_rows.append(
                (lang, sentence)
            )

    # -----------------------------------------------------
    # Shuffle entire multilingual corpus
    # -----------------------------------------------------

    rng.shuffle(all_rows)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["language", "text"]
        )

        writer.writerows(
            all_rows
        )

    # -----------------------------------------------------
    # Verify
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("INDIC MRL CORPUS CREATED")
    print("=" * 60)

    print(
        f"Total sentences: {len(all_rows):,}"
    )

    for lang in LANG_FILES:

        count = sum(
            1
            for x in all_rows
            if x[0] == lang
        )

        print(
            f"{lang}: {count:,}"
        )

    print()
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()