# LocalML website

Static landing page. No framework, no build step - just open `index.html` in a browser.

## Structure

```
website/
├── index.html         # single-page landing
├── styles.css         # design tokens + sections
├── script.js          # copy-to-clipboard, platform highlight, scroll effects
├── install.sh         # macOS/Linux installer  (curl -fsSL .../install.sh | sh)
├── install.ps1        # Windows installer       (irm .../install.ps1 | iex)
├── assets/
│   └── favicon.svg    # LocalML constellation mark
└── README.md          # this file
```

## Install scripts

`install.sh` and `install.ps1` are served as static files from the site root, so
the landing page can advertise a one-liner:

```
# Windows
irm https://localml.app/install.ps1 | iex
# macOS / Linux
curl -fsSL https://localml.app/install.sh | sh
```

Each script **requires an existing Python 3.10+** (it does *not* install Python -
if it's missing the script prints where to get it and stops), then bootstraps
pipx and runs `pipx install localml` (server only). The inference stack
(PyTorch + transformers) is installed **inside the app** on first launch, once
the user picks CPU or GPU - so the script stays fast and hardware-agnostic.

> The scripts and the page hard-code `https://localml.app`. If you deploy to a
> different domain, find-and-replace that host in `install.sh`, `install.ps1`,
> and `index.html` (hero command, the `#install` one-liners, and `script.js`'s
> Windows override). Serving over **HTTPS** is required for `| iex` / `| sh`.

## Local preview

```bash
# any static server works
cd website
python -m http.server 8080
# or: npx serve
```

Open `http://localhost:8080`.