import requests, os, re

os.makedirs("assets", exist_ok=True)
base = "https://www.avantikauniversity.edu.in/"

# Get full page and find all logo/emblem images
r = requests.get(base, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
imgs = re.findall(r'["\'](/assets/[^"\']*?\.(?:png|svg|jpg|webp))["\']', r.text, re.I)
print("All asset images:")
for i in set(imgs):
    if any(k in i.lower() for k in ["logo", "emblem", "icon", "symbol", "crest"]):
        print(" [LOGO]", i)
    else:
        print("       ", i)

# Also try known paths for full/stacked logo
for path in ["assets/vendor/images/logo-full.svg",
             "assets/vendor/images/logo-white.svg",
             "assets/vendor/images/logo_full.png",
             "assets/vendor/images/logo-stacked.svg",
             "assets/images/logo.svg",
             "assets/vendor/images/au-logo.svg",
             "assets/vendor/images/logo.png"]:
    try:
        resp = requests.get(base + path, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if resp.status_code == 200:
            print(f"FOUND: {path} ({len(resp.content)} bytes)")
    except:
        pass
