# Use Transformers for inference
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    engine="transformers",
)
result = ocr.predict("./general_ocr_002.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")