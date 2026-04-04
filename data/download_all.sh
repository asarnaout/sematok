#!/bin/bash
# Download corpora for all supported languages sequentially.
# Runs each language one at a time to avoid cloning race conditions
# when multiple languages share the same data/repos/ directory.
#
# Usage: bash data/download_all.sh
#        bash data/download_all.sh csharp python  # specific languages only

set -eo pipefail

if [ $# -gt 0 ]; then
    LANGUAGES=("$@")
else
    LANGUAGES=($(python -c "from sematok.languages import available_languages; print(' '.join(available_languages()))"))
fi

LOG_DIR="data/download_logs"
mkdir -p "$LOG_DIR"

FAILED=()
SUCCEEDED=()

echo "Languages to download: ${LANGUAGES[*]}"

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "================================================================="
    echo "  DOWNLOADING: $lang"
    echo "================================================================="
    echo ""

    log_file="$LOG_DIR/${lang}.log"
    success=false

    for attempt in 1 2; do
        echo "[Attempt $attempt/2] python -m data.download --language $lang"
        if python -m data.download --language "$lang" 2>&1 | tee "$log_file"; then
            success=true
            break
        else
            echo ""
            echo "WARNING: $lang attempt $attempt failed (exit code $?)"
            if [ "$attempt" -eq 1 ]; then
                echo "Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done

    if $success; then
        SUCCEEDED+=("$lang")
        echo ""
        echo "OK: $lang download complete"
    else
        FAILED+=("$lang")
        echo ""
        echo "FAILED: $lang after 2 attempts. See $log_file"
    fi
done

echo ""
echo "================================================================="
echo "  DOWNLOAD SUMMARY"
echo "================================================================="
echo ""

for lang in "${LANGUAGES[@]}"; do
    ext=$(python -c "from sematok.languages import get_language; print(get_language('$lang').file_extension)")
    raw_dir="data/raw_${lang}"
    if [ -d "$raw_dir" ]; then
        count=$(find "$raw_dir" -maxdepth 1 -name "*${ext}" | wc -l)
        echo "  $lang: $count files in $raw_dir"
    else
        echo "  $lang: MISSING - $raw_dir does not exist!"
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL DOWNLOADS COMPLETED SUCCESSFULLY"
else
    echo "FAILURES: ${FAILED[*]}"
    echo "Check logs in $LOG_DIR/"
fi
