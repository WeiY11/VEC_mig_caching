#!/usr/bin/env python3
"""提取 Benchmarks 目录下的论文 PDF 内容"""
import pdfplumber
from pathlib import Path

papers = [
    ("Nath和Wu - 2020 - Dynamic Computation Offloading and Resource Allocation for Multi-user Mobile Edge Computing.pdf", "Nath_Wu_2020"),
    ("Zhang 等 - 2023 - RoNet Toward Robust Neural Assisted Mobile Network Configuration.pdf", "Zhang_RoNet_2023"),
    ("Liu和Cao - 2022 - Deep learning video analytics through online learning based edge computing.pdf", "Liu_Cao_2022"),
    ("Wang 等 - 2025 - Joint task offloading and migration optimization in UAV-enabled dynamic MEC networks.pdf", "Wang_2025"),
]

output_dir = Path("D:/VEC_mig_caching/Benchmarks/paper_extracts")
output_dir.mkdir(exist_ok=True)

for pdf_name, short_name in papers:
    pdf_path = Path(f"D:/VEC_mig_caching/Benchmarks/{pdf_name}")
    if not pdf_path.exists():
        print(f"NOT FOUND: {pdf_name}")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing: {short_name}")
    print('='*60)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    all_text.append(f"--- Page {i+1} ---\n{text}")
            
            full_text = "\n\n".join(all_text)
            
            # Save to file
            out_path = output_dir / f"{short_name}.txt"
            out_path.write_text(full_text, encoding='utf-8')
            print(f"Saved to: {out_path}")
            
            # Print first 3000 chars
            print(full_text[:3000])
            print("\n... (truncated)")
            
    except Exception as e:
        print(f"ERROR: {e}")
