#!/usr/bin/env bash
set -euo pipefail

# pandoc wrapper for Chinese/English-friendly rendering (no PingFang by default)
# - Core arg: input markdown file (e.g., README.md). May appear anywhere.
# - If core arg missing: default to README.md in current dir.
# - Never overwrite outputs: adds timestamp suffix when needed.
#
# Usage:
#   ./render.sh [INPUT.md]
#     [--outdir DIR] [--title TITLE] [--css PATH]
#     [--html-only | --pdf-only]
#     [--cjkfont NAME] [--mainfont NAME] [--monofont NAME]
#     [--margin VAL] [--no-toc] [--no-embed] [-n|--dry-run]
#     [--no-color]
#
# Examples:
#   ./render.sh                         # render README.md (HTML+PDF)
#   ./render.sh notes.md --outdir out   # render notes.md to out/
#   ./render.sh -n notes.md --pdf-only  # dry-run, only PDF (XeLaTeX)

# --- locale: try to avoid warnings; prefer C / en_US.UTF-8 if available ---
maybe_set_locale() {
  if command -v locale >/dev/null 2>&1; then
    if [[ -z "${LC_ALL:-}" ]]; then
      for L in en_US.UTF-8 C.UTF-8 C; do
        if locale -a 2>/dev/null | grep -qx "$L"; then
          export LC_ALL="$L" LANG="$L"; return
        fi
      done
      export LC_ALL=C LANG=C
    fi
  fi
}
maybe_set_locale

# --- colors ---
COLOR=1
[[ -t 1 ]] || COLOR=0
[[ "${NO_COLOR:-}" != "" ]] && COLOR=0
NO_COLOR_FLAG=0

if [[ $COLOR -eq 1 ]]; then
  C_INFO='\033[1;34m'  # bold blue
  C_WARN='\033[1;33m'  # bold yellow
  C_ERR='\033[1;31m'   # bold red
  C_TAG='\033[1;37m'   # bright tag frame
  C_RST='\033[0m'
else
  C_INFO=''; C_WARN=''; C_ERR=''; C_TAG=''; C_RST=''
fi

info()  { printf "%b[%bINFO%b]%b %s\n" "$C_TAG" "$C_INFO" "$C_TAG" "$C_RST" "$*"; }
warn()  { printf "%b[%bWARN%b]%b %s\n" "$C_TAG" "$C_WARN" "$C_TAG" "$C_RST" "$*"; }
error() { printf "%b[%bERR %b]%b %s\n" "$C_TAG" "$C_ERR" "$C_TAG" "$C_RST" "$*" 1>&2; exit 1; }

has_cmd() { command -v "$1" >/dev/null 2>&1; }

input_md=""

# defaults
outdir="out"
title=""
css_path=""
do_html=1
do_pdf=1
toc=1
embed=1
dryrun=0
cjkfont="Songti SC"
mainfont="Times New Roman"
monofont="Menlo"
margin="1in"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--outdir)    outdir="$2"; shift 2;;
    --title)        title="$2"; shift 2;;
    --css)          css_path="$2"; shift 2;;
    --html-only)    do_html=1; do_pdf=0; shift;;
    --pdf-only)     do_html=0; do_pdf=1; shift;;
    --no-color)     COLOR=0; NO_COLOR_FLAG=1; shift;;
    --cjkfont)      cjkfont="$2"; shift 2;;
    --mainfont)     mainfont="$2"; shift 2;;
    --monofont)     monofont="$2"; shift 2;;
    --margin)       margin="$2"; shift 2;;
    --no-toc)       toc=0; shift;;
    --no-embed)     embed=0; shift;;
    -n|--dry-run)   dryrun=1; shift;;
    -h|--help)
      sed -n '1,70p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    --) shift; break;;
    -*) error "Unknown option: $1";;
    *)  # first non-option is input
      if [[ -z "$input_md" ]]; then input_md="$1"; shift; else error "Multiple inputs not supported: $1"; fi ;;
  esac
done

# remaining args (after --)
if [[ -z "$input_md" && $# -gt 0 ]]; then
  input_md="$1"; shift || true
fi

# Resolve input markdown
if [[ -z "$input_md" ]]; then
  if [[ -f README.md ]]; then
    input_md="README.md"
  else
    # Fallback: if there is exactly one .md file in CWD, use it
    MD_FILES=( $(find . -maxdepth 1 -type f -name '*.md' -print | sed 's/^\.\///g' | tr '\n' ' ') )
    if [[ ${#MD_FILES[@]} -eq 1 && -n "${MD_FILES[0]:-}" ]]; then
      input_md="${MD_FILES[0]}"
      info "Auto-selected single Markdown in directory: $input_md"
    else
      if [[ ${#MD_FILES[@]} -eq 0 || -z "${MD_FILES[0]:-}" ]]; then
        error "No input provided and no Markdown files found. Please pass a Markdown file."
      else
        warn "Multiple Markdown files found:";
        for f in "${MD_FILES[@]}"; do [[ -n "$f" ]] && printf "  - %s\n" "$f"; done
        error "Ambiguous. Please pass the file you want to render explicitly."
      fi
    fi
  fi
fi

[[ -f "$input_md" ]] || error "输入文件不存在: $input_md"

# Ensure tools
has_cmd pandoc || error "Missing pandoc. Please install pandoc first."
if [[ $do_pdf -eq 1 ]]; then
  has_cmd xelatex || warn "xelatex not found. Will generate HTML only." && { has_cmd xelatex || do_pdf=0; }
fi

# Prepare output dir
mkdir -p "$outdir"

base_name=$(basename "$input_md")
stem="${base_name%.*}"

# Default title from stem if not provided
if [[ -z "$title" ]]; then
  title="$stem"
fi

# Default CSS path: outdir/style.css (create if missing)
if [[ -z "$css_path" ]]; then
  css_path="$outdir/style.css"
fi

if [[ ! -f "$css_path" ]]; then
  info "Create default CSS: $css_path"
  cat > "$css_path" <<'CSS'
/* Chinese-friendly typography, avoid PingFang */
html, body {
  font-family: 'Times New Roman', 'Songti SC', 'Heiti SC', 'Noto Serif CJK SC', serif;
  color: #111; background: #fff; line-height: 1.65; font-size: 16px;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
}
h1, h2, h3, h4 { font-weight: 600; line-height: 1.3; }
code, pre { font-family: Menlo, Monaco, Consolas, 'Noto Sans Mono CJK SC', monospace; font-size: 0.95em; }
pre { padding: 0.8em; background: #f6f8fa; border-radius: 6px; overflow: auto; }
blockquote { color: #555; border-left: 4px solid #e5e5e5; padding: 0.4em 1em; margin: 0.8em 0; }
table { border-collapse: collapse; }
th, td { border: 1px solid #e5e5e5; padding: 0.4em 0.6em; }
CSS
fi

# Compute non-overwriting output path
next_out() {
  local path="$1"
  if [[ ! -e "$path" ]]; then printf '%s\n' "$path"; return; fi
  local dir base ext ts candidate i
  dir=$(dirname "$path"); base=$(basename "$path")
  ext="${base##*.}"; base="${base%.*}"
  ts=$(date +%Y%m%d-%H%M%S)
  candidate="$dir/${base}_$ts.$ext"
  if [[ ! -e "$candidate" ]]; then printf '%s\n' "$candidate"; return; fi
  i=1
  while [[ -e "$dir/${base}_${ts}_$i.$ext" ]]; do i=$((i+1)); done
  printf '%s\n' "$dir/${base}_${ts}_$i.$ext"
}

# Targets (non-overwriting)
html_target=$(next_out "$outdir/${stem}.html")
pdf_target=$(next_out "$outdir/${stem}.pdf")

# Flags
common_flags=("$input_md" -s --standalone --resource-path=.)
[[ $toc -eq 1 ]] && common_flags+=(--toc)

# HTML
if [[ $do_html -eq 1 ]]; then
  html_flags=(--mathjax -c "$css_path" --metadata "title=$title")
  [[ $embed -eq 1 ]] && html_flags+=(--embed-resources)
  info "HTML  → $html_target"
  if [[ $dryrun -eq 1 ]]; then
    printf 'pandoc %q ' "${common_flags[@]}"; printf '%q ' "${html_flags[@]}"; printf ' -o %q\n' "$html_target"
  else
    pandoc "${common_flags[@]}" "${html_flags[@]}" -o "$html_target"
  fi
fi

# PDF
if [[ $do_pdf -eq 1 ]]; then
  pdf_flags=(--pdf-engine=xelatex
             -V CJKmainfont="$cjkfont"
             -V mainfont="$mainfont"
             -V monofont="$monofont"
             -V geometry:margin="$margin"
             -V linkcolor=blue -V urlcolor=blue -V toccolor=gray)
  info "PDF   → $pdf_target"
  if [[ $dryrun -eq 1 ]]; then
    printf 'pandoc %q ' "${common_flags[@]}"; printf '%q ' "${pdf_flags[@]}"; printf ' -o %q\n' "$pdf_target"
  else
    pandoc "${common_flags[@]}" "${pdf_flags[@]}" -o "$pdf_target"
  fi
fi

info "Done."
