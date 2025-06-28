from laonlp.tokenize import word_tokenize
import unicodedata

# 1) จัดรูป (compose) ให้ 0xECD+0xEB2 ⇒ 0xEB3
def compose_laosara_am(s: str) -> str:
    return s.replace('\u0ecd\u0eb2', '\u0eb3')

# 2) ทางกลับกัน (ถ้าต้องการ)
def decompose_laosara_am(s: str) -> str:
    return s.replace('\u0eb3', '\u0ecd\u0eb2')

# --- ทดสอบ -----------------------------------------------------
text = "ຄຳ ຄໍາ"
print("ก่อน compose:", [hex(ord(c)) for c in text])

text_fixed = compose_laosara_am(text)
print("หลัง compose :", [hex(ord(c)) for c in text_fixed])
print("สตริงหลัง compose:", text_fixed)

# (แนะนำให้ NFC อีกทีเพื่อความชัวร์)
text_fixed = unicodedata.normalize("NFC", text_fixed)

words = word_tokenize(text_fixed)
print(words)
