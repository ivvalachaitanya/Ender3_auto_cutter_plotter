import re

with open("Cut.svg", "r", encoding="utf-8") as f:
    code = f.read()

# Replace display:none in image tags
fixed = re.sub(r'(<image[^>]*style="[^"]*display:\s*)none', r'\1inline', code)

with open("Cut_fixed.svg", "w", encoding="utf-8") as f:
    f.write(fixed)

print("Successfully un-hidden all images in Cut_fixed.svg!")
