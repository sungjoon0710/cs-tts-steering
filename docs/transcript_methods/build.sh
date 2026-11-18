#!/bin/sh
# Rebuild figures and compile the PDF straight into docs/reports/.
# Needs python3 + pandas + matplotlib, and xelatex (MacTeX) with the macOS fonts Helvetica Neue and Kohinoor Devanagari.
set -e
cd "$(dirname "$0")"
python3 figs.py
for i in 1 2; do
  xelatex -interaction=nonstopmode -output-directory=../reports transcript_methods.tex >/dev/null
done
rm -f ../reports/transcript_methods.aux ../reports/transcript_methods.log ../reports/transcript_methods.out
echo "Built docs/reports/transcript_methods.pdf"
