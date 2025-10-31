# DexLang Compiler 

DexLang is a Python-authored compiler for a lightweight, Python-inspired language that emits portable C and optionally links it into native binaries. Version 2 leans into a hacker-console aesthetic with noisy debug logs, drag‑and‑drop compilation, and a growing set of built-ins for scripting, math, and systems hacking.

## Highlights
- **Indentation-based DexLang syntax** with `let`, `fn`, `if/elif/else`, `while`, `for range(...)`, and inline function calls as statements.
- **String-friendly print pipeline** allowing `"Hello " + name + "!"`, automatic number conversions, and helper functions such as `to_string`, `to_number`, `len`, and `input`.
- **Runtime toolkit**: file IO (`readfile`, `writefile`), timers (`sleep`), randomness (`rand`), debugging (`debug`), shell integration (`system`), and math helpers (`sqrt`, `pow`, `abs`).
- **Messy-mode logging** that surfaces `[parse]`, `[gcc]`, `[debug]`, and `[done]` messages for every line the compiler processes.
- **Drag & drop aware**: dropping a `.dex` file onto `compiler.py` (Windows Explorer / macOS Finder / Linux file manager) compiles immediately using the detected file path.

## Repository Layout
```
dexlang/
├── compiler.py        # DexLang → C compiler + build selector
├── runtime.c          # C runtime helpers (IO, strings, math, system calls)
├── examples/
│   ├── hello.dex      # Basic hello-world walk-through
│   └── input_test.dex # Feature-rich sample covering new built-ins
└── out.c / out / out.exe  # Generated artifacts (overwritten on each build)
```

## Requirements
- Python 3.9+ (tested with 3.12)
- GCC toolchain (for native compilation)
- POSIX shell (Linux/macOS) or PowerShell/CMD (Windows) to run commands; drag-and-drop works on explorer/finder too.

## Quick Start
1. Grab the latest DexLang release ZIP for your platform.
2. Unzip it somewhere writable (e.g., `C:\DexLang` or `~/dexlang`).
3. Run the platform setup script once:
   - **Windows:** double-click `install_dexlang.bat` (installs Python and GCC if missing).
   - **Linux/macOS:** run `./install_dexlang.sh` (may prompt for sudo to install packages).
4. Open a terminal in the extracted folder and run the compiler:

```bash
# Direct invocation
python3 compiler.py examples/hello.dex --mode 1     # Raw C
python3 compiler.py examples/input_test.dex --mode 3  # Linux binary

# Manual prompt (no args)
python3 compiler.py
Enter .dex file path: examples/hello.dex

# Drag & drop (Windows/macOS/Linux file managers)
# Drop myscript.dex onto compiler.py to auto-compile without prompts.
```

When compilation succeeds you’ll see log lines similar to:

```
[init] DexLang compiler v2
[load] runtime.c found
[dexlang] compiling /abs/path/examples/hello.dex
[parse] reading file: /abs/path/examples/hello.dex
[parse] 15 lines
[compile] generating out.c
[compile] generated out.c
[build] mode 2
[gcc] running: gcc out.c runtime.c -o out.exe -lm
[gcc] -> ./out.exe
[done] build ok (0.21s)
[done] exit code 0
```

## Language Snapshot
```dex
print("=== DexLang Messy Test ===")

let name = input("Enter your name: ")
print("Hello " + name)

fn add(x, y):
    return x + y

if add(2, 2) == 4:
    print("Math still works.")
elif add(2, 2) > 4:
    print("Math exploded.")
else:
    print("Something broke.")

for i in range(0, 3):
    debug("loop i=", i)
    print("Loop idx " + i)

let path = "dex_temp.txt"
writefile(path, "Name: " + name + "\n")
print("File contents: " + readfile(path))

system("echo DexLang system test")
print("=== End ===")
```

## Build Selector & Modes
After parsing, the compiler prompts (unless `--mode` is specified):

```
Choose output mode:
1 - Generate raw C code only
2 - Compile to Windows .exe
3 - Compile to Linux executable
Enter choice (1/2/3):
```

- **Mode 1** writes `out.c` only.
- **Mode 2** runs `gcc out.c runtime.c -o out.exe -lm`.
- **Mode 3** runs `gcc out.c runtime.c -o out -lm`.

Outputs reuse the same filenames; remove or rename them if you need multiple variants simultaneously.

## Extending DexLang
- Add parser hooks in `compiler.py` alongside matching codegen to introduce new syntax.
- Extend `runtime.c` to surface additional C helper functions (remember to declare prototypes near the top of generated C).
- Drop new feature demos under `examples/` to ensure manual regression coverage and share usage patterns.

## Troubleshooting
- **`error: file not found -> …`**: path is wrong or missing. When invoked without args, an empty entry triggers this immediately.
- **`gcc` related errors**: ensure GCC is installed and discoverable via `PATH`. WSL or MinGW works on Windows.
- **Runtime issues**: review the messy logs; `debug()` writes to stderr prefixed with `[debug]`.

## License
No license specified yet—adapt as needed before publishing. PRs welcome for SPDX licensing or additional platform support.
