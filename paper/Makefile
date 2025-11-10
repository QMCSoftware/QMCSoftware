# cSpell:ignore maintex cleanall wordcount TEXFLAGS bibdata BIBTEX
# =========================================================
# Combined Build System for JOSS Paper
# =========================================================

# Path Configuration (auto-detected, override via environment variables)
CONDA_BASE ?= $(shell if [ -n "$$CONDA_EXE" ]; then dirname $$(dirname $$CONDA_EXE); elif [ -d "$$HOME/miniconda3" ]; then echo "$$HOME/miniconda3"; elif [ -d "$$HOME/anaconda3" ]; then echo "$$HOME/anaconda3"; else echo "/opt/conda"; fi)
CONDA_BIN  := $(CONDA_BASE)/bin
CONDA_SH   := $(CONDA_BASE)/etc/profile.d/conda.sh
PYTHON     ?= python3
TEX_BIN    ?= $(shell if command -v pdflatex >/dev/null 2>&1; then dirname $$(command -v pdflatex); else echo "/usr/local/texlive/2024/bin/universal-darwin"; fi)
TEX        := $(TEX_BIN)/pdflatex
BIBTEX     := $(TEX_BIN)/bibtex

# File definitions
PAPER      := paper.md
BIB        := paper.bib
MD_SRC     := main.md
SRC        := main.tex
LUA_FILTER := include-code-files.lua

# Output files
OUTPUT     := paper.pdf
OUT        := maintex.pdf
OUT2       := main.pdf

# Build options
PANDOC     := pandoc
CROSSREF   := pandoc-crossref
PDF_ENGINE := xelatex
TEXFLAGS   := --shell-escape
PANDOC_FLAGS := --filter $(CROSSREF) --citeproc --pdf-engine=$(PDF_ENGINE)

# =========================================================
# Targets
# =========================================================

.PHONY: all paper clean cleanall install-mac main_pdf maintex_pdf wordcount show-config

all: main_pdf wordcount

paper: $(OUTPUT)

$(OUTPUT): $(PAPER) $(BIB)
	@echo "Building $(OUTPUT)..."
	$(PANDOC) $(PAPER) -o $(OUTPUT) $(PANDOC_FLAGS) --bibliography=$(BIB)
	@echo "✅ Done: $(OUTPUT) generated successfully."

maintex_pdf: refs_all.bib $(MD_SRC) $(LUA_FILTER)
	@echo "Generating main.tex from main.md with natbib..."
	pandoc $(MD_SRC) -o $(SRC) --template=template.latex --lua-filter=include-code-files.lua --natbib --bibliography=refs_all.bib -M colorlinks=true -M linkcolor=blue -M citecolor=blue -M urlcolor=blue -M filecolor=magenta
	@echo "Adding bibliography commands to main.tex..."
	@if ! grep -q '\\bibliography{' $(SRC); then \
		perl -i.bak -pe 's/(\\end\{document\})/\\bibliographystyle{abbrvnat}\n\\bibliography{refs_all}\n$$1/' $(SRC); \
	fi
	@echo "Compiling main.tex with pdflatex..."
	rm -f maintex.bbl maintex.aux
	PATH="$$PATH:$(CONDA_BIN)" $(TEX) -jobname=maintex $(TEXFLAGS) $(SRC)
	if grep -q 'citation\|bibdata' maintex.aux 2>/dev/null; then PATH="$$PATH:$(CONDA_BIN)" $(BIBTEX) maintex; fi
	PATH="$$PATH:$(CONDA_BIN)" $(TEX) -jobname=maintex $(TEXFLAGS) $(SRC)
	PATH="$$PATH:$(CONDA_BIN)" $(TEX) -jobname=maintex $(TEXFLAGS) $(SRC)
	@echo "✅ Done: $(OUT) generated successfully."
	open $(OUT) &

main_pdf: cleanall refs_all.bib $(MD_SRC) $(LUA_FILTER)
	@echo "Building main.pdf from $(MD_SRC) with Pandoc and Lua filter..."
	@if command -v $(CROSSREF) >/dev/null 2>&1; then \
	  EXTRA_FILTER="--filter $(CROSSREF)"; \
	  echo "Using pandoc-crossref"; \
	else \
	  EXTRA_FILTER=""; \
	  echo "pandoc-crossref not found; proceeding without it"; \
	fi; \
	$(PANDOC) $(MD_SRC) -o main.pdf --template=template.latex --lua-filter=$(LUA_FILTER) $$EXTRA_FILTER --citeproc --bibliography=refs_all.bib --csl=joss-simple.csl --pdf-engine=$(PDF_ENGINE)
	@echo "✅ Done: main.pdf generated successfully."
	open main.pdf &

refs_all.bib: FJH25.bib FJHown25.bib choi2025.bib other.bib
	@echo "Generating refs_all.bib..."
	cat $^ > $@
	@echo "Cleaning LaTeX commands from refs_all.bib..."
	@grep -v '^[[:space:]]*\\(def\|providecommand\|newcommand)' $@ > $@.tmp && mv $@.tmp $@
	sed -i.bak '79,$$s/\\HickernellFJ/Hickernell/g' $@
	sed -i.bak -E 's/@(online|software|url)\{/@misc{/g; s/@inproceedings\{/@article{/g' $@
	@echo "Adding author field to @proceedings entries for natbib compatibility..."
	@perl -i.bak2 -pe 'if (/^\@proceedings/) {$$in_proc=1} if ($$in_proc && /^(\s*)editor\s*=\s*\{([^}]+)\},?$$/) {$$ed=$$2; $$sp=$$1; $$_=$$_ . "$${sp}author = {$$ed},\n"; $$in_proc=0}' $@
	sed -i.bak -E 's/Choi, Sou[- ]?Cheng( T\.?)?/Choi, S.-C. T./g' $@
	sed -i.bak -E 's/Dick, Josef/Dick, J./g' $@
	sed -i.bak -E 's/Xiaoqun Wang/Wang, X./g' $@
	sed -i.bak -E 's/((A.|Art) B. Owen)|(Owen, Art B)/Owen, A. B./g' $@
	@echo "✅ refs_all.bib generated successfully."

clean:
	find . -type f \( -name '*.aux' -o -name '*.fdb_latexmk' -o -name '*.fls' -o -name '*.log' -o -name '*.out' -o -name '*.synctex.gz' -o -name '*.toc' -o -name '*.bbl' -o -name '*.blg' -o -name '*.awk' -o -name '*.pyg' \) -delete

cleanall: clean
	rm -f $(OUTPUT) $(OUT) $(OUT2) main.pdf refs_all.bib*

install-mac:
	@echo "Installing dependencies (Pandoc, pandoc-crossref, TeX Live)..."
	brew install pandoc pandoc-crossref || true
	@echo "Checking for existing TeX Live installations..."
	@if brew list --cask basictex >/dev/null 2>&1; then \
		echo "✅ BasicTeX is already installed"; \
	elif brew list --cask mactex-no-gui >/dev/null 2>&1; then \
		echo "✅ MacTeX-no-GUI is already installed"; \
	elif brew list --cask mactex >/dev/null 2>&1; then \
		echo "✅ MacTeX is already installed"; \
	else \
		echo "Installing MacTeX-no-GUI..."; \
		brew install --cask mactex-no-gui || echo "⚠️  MacTeX installation failed. You may need to install it manually from https://www.tug.org/mactex/"; \
	fi
	@echo "Configuring TeX Live package manager..."
	@if command -v tlmgr >/dev/null 2>&1; then \
		TL_YEAR=$$(tlmgr --version 2>/dev/null | awk '/TeX Live/ {print $$4}' | head -n1); \
		if [ -n "$$TL_YEAR" ]; then \
			echo "Detected TeX Live $$TL_YEAR"; \
			echo "Setting repository to CTAN mirror..."; \
			sudo tlmgr option repository https://mirror.ctan.org/systems/texlive/tlnet || \
			sudo tlmgr option repository ftp://tug.org/texlive/tlnet || \
			sudo tlmgr option repository https://ctan.org/tex-archive/systems/texlive/tlnet; \
			echo "Updating tlmgr..."; \
			sudo tlmgr update --self || echo "⚠️  tlmgr update failed (may need manual intervention)"; \
			echo "Installing TeX Live fonts collection..."; \
			sudo tlmgr install collection-fontsrecommended || echo "⚠️  Font installation failed"; \
		else \
			echo "⚠️  Could not detect TeX Live version; skipping tlmgr configuration."; \
		fi \
	else \
		echo "⚠️  tlmgr not found. Please ensure MacTeX is properly installed."; \
		echo "   You may need to add it to your PATH: export PATH=/usr/local/texlive/2024/bin/universal-darwin:\$$PATH"; \
	fi
	@echo "✅ Installation complete.

show-config:
	@echo "=== Detected Configuration ==="
	@echo "CONDA_BASE: $(CONDA_BASE)"
	@echo "CONDA_BIN:  $(CONDA_BIN)"
	@echo "PYTHON:     $(PYTHON)"
	@echo "TEX_BIN:    $(TEX_BIN)"
	@echo ""
	@echo "To override: make VARIABLE=value target"
	@echo "Example: make CONDA_BASE=/opt/conda main_pdf"

wordcount:
	# Count words in the Markdown source file, excluding YAML header, comments, citations, 
	# figure paths/attributes (but keeping captions), standalone punctuation, and everything after the "Acknowledgements" section.
	@echo "Counting words in $(MD_SRC) (exclude YAML header, comments, citations, figure syntax, standalone punctuation, and everything after Acknowledgements)..."
	@awk 'BEGIN{in_comment=0; in_yaml=1} /^---$$/ && in_yaml && NR>1{in_yaml=0; next} in_yaml{next} /^[[:space:]]*<!--/{ if ($$0 ~ /-->/) next; in_comment=1; next } in_comment{ if ($$0 ~ /-->/) in_comment=0; next } /^# (Acknowledgements|References)$$/{ exit } { print }' $(MD_SRC) | tr '\n' '\r' | sed -E 's/!\[/ /g; s/\]\([^)]*\)\{[^}]*\}//g; s/\[@[^]]*\]//g; s/\\label\{[^}]*\}//g; s/ ([,.])/\1/g; s/\r([,.])/\1/g' | tr '\r' '\n' | tr -cd "\11\12\15\40-\176" | tr -s ' ' '\n' | grep -vE '^[[:punct:]]+$$' | wc -l

	# Generate a temporary Markdown file (tmp.md) for verification purposes, 
	# applying the same filtering rules as the word count.
	@echo "Generating tmp.md for verification..."
	@awk 'BEGIN{in_comment=0; in_yaml=1} /^---$$/ && in_yaml && NR>1{in_yaml=0; next} in_yaml{next} /^[[:space:]]*<!--/{ if ($$0 ~ /-->/) next; in_comment=1; next } in_comment{ if ($$0 ~ /-->/) in_comment=0; next } /^# (Acknowledgements|References)$$/{ exit } { print }' $(MD_SRC) | tr '\n' '\r' | sed -E 's/!\[/ /g; s/\]\([^)]*\)\{[^}]*\}//g; s/\[@[^]]*\]//g; s/\\label\{[^}]*\}//g; s/ ([,.])/\1/g; s/\r([,.])/\1/g' | tr '\r' '\n' > tmp.md

