#!/usr/bin/env bash
# Install the SMuFL music fonts Verovio engraves with, for the rasterizer.
#
# Notation glyphs come from Verovio's own glyph data, but SMuFL characters inside
# SVG text (metronome marks, figured bass...) are drawn by cairosvg from an
# installed font named after the engraving font ("Leland", "Petaluma"...).
# Without it they render in a fallback font, or not at all.
#
# Fonts come from the Verovio release matching the installed verovio, so text
# glyphs match the engraved ones; Petaluma from Steinberg. All SIL OFL 1.1.
#
#   scripts/install-fonts.sh             install the missing fonts
#   scripts/install-fonts.sh --force     (re)install all of them
#   FONT_DIR=/usr/local/share/fonts/musicset sudo -E scripts/install-fonts.sh   system-wide

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Homebrew's cairo picks text fonts through CoreText, which never reads the XDG
# fonts dir; ~/Library/Fonts is seen by both CoreText and fontconfig
if [ "$(uname -s)" = Darwin ]; then
    FONT_DIR="${FONT_DIR:-$HOME/Library/Fonts/musicset}"
else
    FONT_DIR="${FONT_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/fonts/musicset}"
fi
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

for tool in curl fc-cache fc-list fc-match; do
    command -v "$tool" >/dev/null || { echo "missing $tool (install curl and fontconfig)" >&2; exit 1; }
done

# Verovio tag of the installed toolkit, e.g. 6.3.0-425dd7b -> version-6.3.0
if [ -z "${VEROVIO_TAG:-}" ]; then
    version=$(cd "$ROOT" && uv run --quiet python -c \
        "import verovio; print(verovio.toolkit().getVersion().split('-')[0])" 2>/dev/null || true)
    VEROVIO_TAG="version-${version:-6.3.0}"
fi
VEROVIO="https://raw.githubusercontent.com/rism-digital/verovio/$VEROVIO_TAG/fonts"
PETALUMA="https://raw.githubusercontent.com/steinbergmedia/petaluma/petaluma-1.065/redist/otf"

# a case, not an associative array: macOS ships bash 3.2
font_url() {
    case "$1" in
        Bravura) echo "$VEROVIO/Bravura/Bravura.otf" ;;
        Gootville) echo "$VEROVIO/Gootville/Gootville.otf" ;;
        Leipzig) echo "$VEROVIO/Leipzig/Leipzig.ttf" ;;
        Leland) echo "$VEROVIO/Leland/Leland.otf" ;;
        Petaluma) echo "$PETALUMA/Petaluma.otf" ;;
    esac
}

mkdir -p "$FONT_DIR"
echo "Installing into $FONT_DIR (Verovio fonts from $VEROVIO_TAG)"
for name in Bravura Gootville Leipzig Leland Petaluma; do
    existing=$(fc-list ":family=$name" file | head -n1)
    if [ -n "$existing" ] && [ "$FORCE" = 0 ]; then
        echo "  $name: already installed (${existing%%:*})"
        continue
    fi
    url=$(font_url "$name")
    target="$FONT_DIR/${url##*/}"
    curl -fsSL --retry 3 -o "$target.part" "$url"
    mv "$target.part" "$target"
    echo "  $name: installed $(basename "$target")"
done

fc-cache -f "$FONT_DIR" >/dev/null

status=0
for name in Bravura Gootville Leipzig Leland Petaluma; do
    family=$(fc-match -f '%{family}' "$name")
    if [[ "$family" != *"$name"* ]]; then
        echo "  $name: NOT resolved (fontconfig falls back to '$family')" >&2
        status=1
    fi
done

# Verovio writes text in "Times"; boxes are measured with whatever it resolves to
times=$(fc-match -f '%{family}' Times)
case "$times" in
    *"Nimbus Roman"* | *Times* | *"Liberation Serif"* | *"TeX Gyre Termes"* | *FreeSerif*)
        echo "Text font: Times -> $times" ;;
    *)
        echo "Text font: Times -> $times, not Times-compatible (install fonts-urw-base35)" >&2
        status=1 ;;
esac

[ "$status" = 0 ] && echo "All music fonts resolve."
exit "$status"
