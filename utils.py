import csv
import os
from datetime import datetime
import numpy as np

class ResultsManager:
    def __init__(self):
        self.results = None
        
    def save_to_csv(self, results, output_path=None, video_name=""):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"analiz_sonuclari_{timestamp}.csv"
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                writer.writerow(['# Eritrosit Hızı Analiz Sonuçları (UZD + Hough Transform)'])
                writer.writerow(['# Tarih:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow(['# Video:', video_name])
                writer.writerow([])
                
                writer.writerow(['ISTATISTIK', 'DEGER (um/s)'])
                writer.writerow(['Ortalama', f"{results.get('mean', 0):.2f}"])
                writer.writerow(['Medyan', f"{results.get('median', 0):.2f}"])
                writer.writerow(['Standart Sapma', f"{results.get('std', 0):.2f}"])
                writer.writerow(['Minimum', f"{results.get('min', 0):.2f}"])
                writer.writerow(['Maksimum', f"{results.get('max', 0):.2f}"])
                writer.writerow([])
                
                writer.writerow(['PERSANTILLER', 'DEGER (um/s)'])
                writer.writerow(['25. Persantil', f"{results.get('p25', 0):.2f}"])
                writer.writerow(['50. Persantil (Medyan)', f"{results.get('p50', 0):.2f}"])
                writer.writerow(['75. Persantil', f"{results.get('p75', 0):.2f}"])
                writer.writerow(['90. Persantil', f"{results.get('p90', 0):.2f}"])
                writer.writerow(['95. Persantil', f"{results.get('p95', 0):.2f}"])
                writer.writerow([])
                
                writer.writerow(['OLCUM SAYILARI', 'DEGER'])
                writer.writerow(['Gecerli Olcum (n_valid)', results.get('n_valid', 0)])
                writer.writerow(['Aliasing Supheli (n_alias)', results.get('n_alias', 0)])
                writer.writerow(['Toplam Olcum', results.get('n_total', 0)])
                writer.writerow([])
                
                analysis_info = results.get('analysis_info', {})
                writer.writerow(['ANALIZ BILGILERI', 'DEGER'])
                writer.writerow(['Arka Plan Modu', analysis_info.get('background_mode', 'bilinmiyor')])
                writer.writerow(['OD Donusumu', analysis_info.get('od_used', False)])
                writer.writerow([])
                
                speeds = results.get('all_speeds', [])
                if speeds:
                    writer.writerow(['TUM HIZ DEGERLERI (um/s)'])
                    for i, speed in enumerate(speeds, 1):
                        writer.writerow([f'Olcum {i}', f'{speed:.2f}'])
            
            return output_path
        except Exception as e:
            print(f"CSV kaydetme hatası: {e}")
            return None
    
    def format_results_for_display(self, results):
        if results is None:
            return "Sonuç yok"
        
        lines = []
        lines.append(f"Ortalama: {results.get('mean', 0):.1f} um/s")
        lines.append(f"Medyan: {results.get('median', 0):.1f} um/s")
        lines.append(f"SD: {results.get('std', 0):.1f} um/s")
        lines.append(f"Min-Max: {results.get('min', 0):.1f} - {results.get('max', 0):.1f} um/s")
        lines.append(f"Gecerli: {results.get('n_valid', 0)} | Aliasing: {results.get('n_alias', 0)}")
        
        return " | ".join(lines)


class HistogramGenerator:
    def __init__(self):
        pass
    
    def create_histogram(self, speeds, output_path=None, show=True):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib yuklu degil")
            return None
        
        if len(speeds) == 0:
            return None
        
        speeds_arr = np.array(speeds)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins, patches = ax.hist(speeds_arr, bins=30, color='#2196F3', 
                                    edgecolor='white', alpha=0.7)
        
        mean_val = np.mean(speeds_arr)
        median_val = np.median(speeds_arr)
        p25 = np.percentile(speeds_arr, 25)
        p75 = np.percentile(speeds_arr, 75)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Ortalama: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Medyan: {median_val:.1f}')
        ax.axvline(p25, color='orange', linestyle=':', linewidth=1.5, label=f'P25: {p25:.1f}')
        ax.axvline(p75, color='orange', linestyle=':', linewidth=1.5, label=f'P75: {p75:.1f}')
        
        ax.set_xlabel('Hiz (um/s)', fontsize=12)
        ax.set_ylabel('Frekans', fontsize=12)
        ax.set_title('Eritrosit Hizi Dagilimi', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        textstr = f'n = {len(speeds_arr)}\nSD = {np.std(speeds_arr):.1f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.85, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"histogram_{timestamp}.png"
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_histogram_bytes(self, speeds):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from io import BytesIO
        except ImportError:
            return None
        
        if len(speeds) == 0:
            return None
        
        speeds_arr = np.array(speeds)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(speeds_arr, bins=25, color='#2196F3', edgecolor='white', alpha=0.7)
        
        mean_val = np.mean(speeds_arr)
        median_val = np.median(speeds_arr)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Ort: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Med: {median_val:.1f}')
        
        ax.set_xlabel('Hiz (um/s)')
        ax.set_ylabel('Frekans')
        ax.set_title('Eritrosit Hizi Dagilimi')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        return buf.getvalue()


class QualityChecker:
    def __init__(self, fps, pixel_to_um=1.832):
        self.fps = fps
        self.pixel_to_um = pixel_to_um
        
    def calculate_nyquist_limit(self, roi_height):
        max_displacement_per_frame = roi_height / 2
        max_speed = max_displacement_per_frame * self.pixel_to_um * self.fps
        return max_speed
    
    def check_aliasing(self, speed, roi_height):
        nyquist_limit = self.calculate_nyquist_limit(roi_height)
        
        if speed > nyquist_limit:
            return {
                'is_aliasing': True,
                'nyquist_limit': nyquist_limit,
                'speed': speed,
                'ratio': speed / nyquist_limit
            }
        
        return {
            'is_aliasing': False,
            'nyquist_limit': nyquist_limit,
            'speed': speed,
            'ratio': speed / nyquist_limit
        }
    
    def filter_aliased_speeds(self, speeds, roi_height):
        valid = []
        aliased = []
        
        for speed in speeds:
            result = self.check_aliasing(speed, roi_height)
            if result['is_aliasing']:
                aliased.append(speed)
            else:
                valid.append(speed)
        
        return valid, aliased

