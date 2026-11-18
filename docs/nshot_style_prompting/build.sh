#!/bin/sh
# Rebuild figures/tables and compile the PDF straight into docs/reports/.
# Needs python3 + pandas + matplotlib, and xelatex (MacTeX) with the macOS font Helvetica Neue.
set -e
cd "$(dirname "$0")"
python3 figs.py
for i in 1 2; do
  xelatex -interaction=nonstopmode -output-directory=../reports nshot_style_prompting.tex >/dev/null
done
rm -f ../reports/nshot_style_prompting.aux ../reports/nshot_style_prompting.log ../reports/nshot_style_prompting.out
echo "Built docs/reports/nshot_style_prompting.pdf"
