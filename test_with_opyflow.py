import sys
sys.path.append("benzer_projeler/opyflow/src")

try:
    import opyf
    
    print("OpyFlow ile test")
    print("=" * 50)
    
    video_path = "10_hasta_sl.avi"
    
    analyzer = opyf.videoAnalyzer(video_path)
    
    analyzer.extractGoodFeaturesAndDisplacements()
    
    print("Analiz tamamlandi!")
    print("OpyFlow sonuclari bizim sistemle karsilastirilabilir")
    
except ImportError as e:
    print(f"OpyFlow yuklu degil: {e}")
    print("Yuklemek icin: pip install opyf")
except Exception as e:
    print(f"Hata: {e}")

