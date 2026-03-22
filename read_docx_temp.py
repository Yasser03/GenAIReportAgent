import sys
try:
    import docx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    import docx

doc = docx.Document(sys.argv[1])
for para in doc.paragraphs:
    print(para.text)
