#Programm to scan pdf-files and make them machine-readable
import os 
import ocrmypdf 

from pathlib import Path

outpath = "./files/GOOD"

inpath = "./files/list"

Path(outpath).mkdir(parents=True, exist_ok=True)

for filename in os.listdir(inpath):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(inpath, filename)
            output_path = os.path.join(outpath, filename)
            
            print(f"Processing: {filename}")
            try:
                # Apply OCR and save to output folder
                ocrmypdf.ocr(
                    deskew=True,
                    clean=True,
                    language="rus+eng",
                    input_file=input_path,
                    output_file=output_path,
                    skip_text=True,
                    force_ocr=False, 
                    optimize=1,          # Optimize PDF size
                    progress_bar=True  
                )
                print(f"✓ Success: {filename}")
            except ocrmypdf.exceptions.PriorOcrFoundError:
                print(f"⏩ Skipped (already has OCR): {filename}")
            except Exception as e:
                print(f"❌ Failed: {filename} - {str(e)}")