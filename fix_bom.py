"""fix_bom.py -- run once: python fix_bom.py"""
files = ["requirements.txt", "agents/README.md", "INTEGRATION.md"]
for fn in files:
    with open(fn, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(fn, "wb") as f:
            f.write(data[3:])
        print(f"Fixed BOM in {fn}")
    else:
        print(f"{fn} already clean")