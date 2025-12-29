import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QFrame, QGridLayout, QScrollArea, QInputDialog,
                            QProgressBar, QDialog, QMessageBox, QSlider)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor

os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
cv2.setLogLevel(0)

from analysis_engine import SpaceTimeDiagramAnalyzer
from utils import ResultsManager, HistogramGenerator
from sam_processor import SAMVesselSegmenter, LightweightVesselSegmenter, create_vessel_segmenter, HybridVesselTracker, create_vessel_tracker

class AnalysisThread(QThread):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, video_path, roi, fps, pixel_to_um, centerline_points=None):
        super().__init__()
        self.video_path = video_path
        self.roi = roi
        self.fps = fps
        self.pixel_to_um = pixel_to_um
        self.centerline_points = centerline_points
    
    def run(self):
        try:
            analyzer = SpaceTimeDiagramAnalyzer(self.fps, self.pixel_to_um)
            results = analyzer.analyze_video(
                self.video_path, 
                self.roi,
                progress_callback=self.update_progress,
                centerline_points=self.centerline_points
            )
            
            if results:
                self.finished.emit(results)
            else:
                self.error.emit("Analysis result could not be obtained")
        except Exception as e:
            self.error.emit(str(e))
    
    def update_progress(self, current, total, message):
        self.progress.emit(current, total, message)


class ModernButton(QPushButton):
    def __init__(self, text, primary=False):
        super().__init__(text)
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: 600;
                    min-width: 90px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
                QPushButton:disabled {
                    background-color: #BDBDBD;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    color: #2196F3;
                    border: 2px solid #2196F3;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: 600;
                    min-width: 90px;
                }
                QPushButton:hover {
                    background-color: #E3F2FD;
                }
                QPushButton:pressed {
                    background-color: #BBDEFB;
                }
                QPushButton:disabled {
                    background-color: #F5F5F5;
                    color: #BDBDBD;
                    border-color: #E0E0E0;
                }
            """)


class InfoCard(QFrame):
    def __init__(self, title, value, unit=""):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(8, 8, 8, 8)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #757575; font-size: 14px; font-weight: 500;")
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: #212121; font-size: 23px; font-weight: 700;")
        
        unit_label = QLabel(unit)
        unit_label.setStyleSheet("color: #9E9E9E; font-size: 13px; font-weight: 600;")
        
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(unit_label)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_value(self, value):
        self.value_label.setText(value)


class ImageViewerDialog(QDialog):
    def __init__(self, title, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        
        if len(image.shape) == 2:
            h, w = image.shape
            q_img = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(780, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        
        layout.addWidget(label)
        
        close_btn = ModernButton("Close", primary=True)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setMinimumSize(400, 200)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title_label = QLabel("Erytroscope")
        title_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #212121;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        info_label = QLabel("Developed by Ibrahim UNAL\nUnder the supervision of Prof. Dr. Ugur AKSU")
        info_label.setStyleSheet("font-size: 14px; color: #616161; font-weight: 500; line-height: 1.6;")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        close_btn = ModernButton("Close", primary=True)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)


class EritrosidAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.cap = None
        self.first_frame = None
        self.current_frame = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = None
        self.pixel_to_um = 1.832
        self.analysis_results = None
        self.results_manager = ResultsManager()
        self.histogram_generator = HistogramGenerator()
        self.analysis_thread = None
        self.is_live_playing = False
        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.update_live_frame)
        self.prev_live_frame = None
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Erytroscope")

        # Ekran boyutunu al ve pencereyi ekranın %85'i olarak ayarla
        screen = QApplication.primaryScreen().geometry()
        window_width = int(screen.width() * 0.85)
        window_height = int(screen.height() * 0.85)

        # Pencereyi ortala
        x = (screen.width() - window_width) // 2
        y = (screen.height() - window_height) // 2
        self.setGeometry(x, y, window_width, window_height)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QLabel {
                color: #212121;
                font-weight: 600;
            }
        """)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        
        title_label = QLabel("Erythrocyte Velocity Analysis")
        title_label.setStyleSheet("font-size: 20px; font-weight: 700; color: #212121;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        method_label = QLabel("STD + Hough Transform")
        method_label.setStyleSheet("font-size: 12px; color: #4CAF50; font-weight: 600; background-color: #E8F5E9; padding: 6px 12px; border-radius: 4px;")
        header_layout.addWidget(method_label)
        
        about_btn = QPushButton("About")
        about_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        about_btn.clicked.connect(self.show_about)
        header_layout.addWidget(about_btn)
        
        exit_btn = QPushButton("Close")
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        exit_btn.clicked.connect(self.close)
        header_layout.addWidget(exit_btn)
        
        main_layout.addWidget(header_frame)
        
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        video_frame = QFrame()
        video_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        video_layout = QVBoxLayout(video_frame)

        self.video_label = QLabel()
        self.video_label.setMinimumHeight(400)
        from PySide6.QtWidgets import QSizePolicy
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #FAFAFA;
                border: 2px dashed #E0E0E0;
                border-radius: 8px;
                color: #9E9E9E;
                font-size: 14px;
            }
        """)
        self.video_label.setText("No video loaded\n\nPlease select a video file")
        video_layout.addWidget(self.video_label, 3)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.load_btn = ModernButton("Load Video", primary=True)
        self.load_btn.clicked.connect(self.load_video)
        button_layout.addWidget(self.load_btn)
        
        self.select_roi_btn = ModernButton("Select Region")
        self.select_roi_btn.clicked.connect(self.select_roi)
        self.select_roi_btn.setEnabled(False)
        button_layout.addWidget(self.select_roi_btn)
        
        self.analyze_btn = ModernButton("Full Analysis", primary=True)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        button_layout.addWidget(self.analyze_btn)
        
        self.live_btn = ModernButton("Live Preview")
        self.live_btn.clicked.connect(self.toggle_live_preview)
        self.live_btn.setEnabled(False)
        button_layout.addWidget(self.live_btn)
        
        button_layout.addStretch()
        video_layout.addLayout(button_layout)
        
        frame_nav_frame = QFrame()
        frame_nav_frame.setStyleSheet("""
            QFrame {
                background-color: #F5F5F5;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        frame_nav_layout = QVBoxLayout(frame_nav_frame)
        frame_nav_layout.setSpacing(5)
        frame_nav_layout.setContentsMargins(5, 5, 5, 5)
        
        self.frame_info_label = QLabel("Frame: 0 / 0")
        self.frame_info_label.setStyleSheet("font-size: 11px; color: #616161; font-weight: 600;")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        frame_nav_layout.addWidget(self.frame_info_label)
        
        nav_controls_layout = QHBoxLayout()
        
        self.prev_frame_btn = QPushButton("◀ Previous")
        self.prev_frame_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #2196F3;
                border: 1px solid #2196F3;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #E3F2FD;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #BDBDBD;
                border-color: #E0E0E0;
            }
        """)
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.prev_frame_btn.setEnabled(False)
        nav_controls_layout.addWidget(self.prev_frame_btn)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #E0E0E0;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #1976D2;
            }
        """)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        self.frame_slider.setEnabled(False)
        nav_controls_layout.addWidget(self.frame_slider, 1)
        
        self.next_frame_btn = QPushButton("Next ▶")
        self.next_frame_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #2196F3;
                border: 1px solid #2196F3;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #E3F2FD;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #BDBDBD;
                border-color: #E0E0E0;
            }
        """)
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)
        nav_controls_layout.addWidget(self.next_frame_btn)
        
        frame_nav_layout.addLayout(nav_controls_layout)
        frame_nav_frame.setMaximumHeight(80)
        video_layout.addWidget(frame_nav_frame, 0)
        self.frame_nav_frame = frame_nav_frame
        self.frame_nav_frame.setVisible(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #E0E0E0;
                height: 20px;
                text-align: center;
                color: #000000;
                font-weight: 600;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 4px;
            }
        """)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        video_layout.addWidget(self.progress_bar)
        
        left_layout.addWidget(video_frame)
        
        # Sağ panel için scroll area oluştur
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setMaximumWidth(310)
        right_scroll.setMinimumWidth(280)
        right_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        
        info_title = QLabel("Video Information")
        info_title.setStyleSheet("font-size: 19px; font-weight: 600; color: #212121; margin-bottom: 10px;")
        info_layout.addWidget(info_title)
        
        self.video_name_label = QLabel("Video: -")
        self.video_name_label.setStyleSheet("font-size: 15px; color: #616161; padding: 8px; font-weight: 600;")
        self.video_name_label.setWordWrap(True)
        info_layout.addWidget(self.video_name_label)
        
        fps_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: -")
        self.fps_label.setStyleSheet("font-size: 17px; color: #616161; padding: 8px; font-weight: 600;")
        fps_layout.addWidget(self.fps_label)
        
        fps_edit_btn = QPushButton("Edit")
        fps_edit_btn.setMaximumWidth(70)
        fps_edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        fps_edit_btn.clicked.connect(self.edit_fps)
        fps_layout.addWidget(fps_edit_btn)
        
        info_layout.addLayout(fps_layout)
        
        scale_layout = QHBoxLayout()
        self.scale_label = QLabel(f"Scale: {self.pixel_to_um} um/pixel")
        self.scale_label.setStyleSheet("font-size: 17px; color: #616161; padding: 8px; font-weight: 600;")
        scale_layout.addWidget(self.scale_label)
        
        scale_edit_btn = QPushButton("Edit")
        scale_edit_btn.setMaximumWidth(70)
        scale_edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        scale_edit_btn.clicked.connect(self.edit_scale)
        scale_layout.addWidget(scale_edit_btn)
        
        info_layout.addLayout(scale_layout)
        
        self.frame_count_label = QLabel("Frame Count: -")
        self.frame_count_label.setStyleSheet("font-size: 17px; color: #616161; padding: 8px; font-weight: 600;")
        info_layout.addWidget(self.frame_count_label)
        
        info_layout.addStretch()
        right_layout.addWidget(info_frame)
        
        results_frame = QFrame()
        results_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        results_layout = QVBoxLayout(results_frame)
        
        results_title = QLabel("Analysis Results")
        results_title.setStyleSheet("font-size: 19px; font-weight: 600; color: #212121; margin-bottom: 15px;")
        results_layout.addWidget(results_title)
        
        self.avg_speed_card = InfoCard("Average Speed", "-", "um/s")
        results_layout.addWidget(self.avg_speed_card)
        
        self.median_speed_card = InfoCard("Median Speed", "-", "um/s")
        results_layout.addWidget(self.median_speed_card)
        
        self.std_speed_card = InfoCard("Standard Deviation", "-", "um/s")
        results_layout.addWidget(self.std_speed_card)
        
        self.min_speed_card = InfoCard("Minimum Speed", "-", "um/s")
        results_layout.addWidget(self.min_speed_card)
        
        self.max_speed_card = InfoCard("Maximum Speed", "-", "um/s")
        results_layout.addWidget(self.max_speed_card)
        
        count_frame = QFrame()
        count_frame.setStyleSheet("""
            QFrame {
                background-color: #F5F5F5;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        count_layout = QHBoxLayout(count_frame)
        
        self.valid_label = QLabel("Valid: -")
        self.valid_label.setStyleSheet("font-size: 15px; color: #4CAF50; font-weight: 600;")
        count_layout.addWidget(self.valid_label)
        
        self.alias_label = QLabel("Aliasing: -")
        self.alias_label.setStyleSheet("font-size: 15px; color: #FF9800; font-weight: 600;")
        count_layout.addWidget(self.alias_label)
        
        results_layout.addWidget(count_frame)
        
        results_layout.addStretch()
        right_layout.addWidget(results_frame)
        
        output_frame = QFrame()
        output_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        output_layout = QVBoxLayout(output_frame)
        
        output_title = QLabel("Outputs")
        output_title.setStyleSheet("font-size: 17px; font-weight: 600; color: #212121; margin-bottom: 10px;")
        output_layout.addWidget(output_title)
        
        self.show_uzd_btn = QPushButton("Show STD")
        self.show_uzd_btn.setStyleSheet("""
            QPushButton {
                background-color: #E3F2FD;
                color: #1976D2;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #BBDEFB;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #BDBDBD;
            }
        """)
        self.show_uzd_btn.clicked.connect(self.show_uzd)
        self.show_uzd_btn.setEnabled(False)
        output_layout.addWidget(self.show_uzd_btn)
        
        self.show_histogram_btn = QPushButton("Show Histogram")
        self.show_histogram_btn.setStyleSheet("""
            QPushButton {
                background-color: #E8F5E9;
                color: #388E3C;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #C8E6C9;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #BDBDBD;
            }
        """)
        self.show_histogram_btn.clicked.connect(self.show_histogram)
        self.show_histogram_btn.setEnabled(False)
        output_layout.addWidget(self.show_histogram_btn)
        
        self.save_csv_btn = QPushButton("Save CSV")
        self.save_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFF3E0;
                color: #E65100;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #FFE0B2;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #BDBDBD;
            }
        """)
        self.save_csv_btn.clicked.connect(self.save_csv)
        self.save_csv_btn.setEnabled(False)
        output_layout.addWidget(self.save_csv_btn)
        
        right_layout.addWidget(output_frame)

        right_layout.addStretch()

        # Sağ paneli scroll area'ya ekle
        right_scroll.setWidget(right_panel)

        content_layout.addWidget(left_panel, 3)
        content_layout.addWidget(right_scroll, 1)
        
        main_layout.addLayout(content_layout)
        
        status_bar = QLabel("Ready")
        status_bar.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 10px 20px;
                border-radius: 8px;
                color: #757575;
                font-size: 12px;
                font-weight: 600;
            }
        """)
        main_layout.addWidget(status_bar)
        self.status_label = status_bar

    def resizeEvent(self, event):
        """Pencere yeniden boyutlandırıldığında videoyu yeniden çiz"""
        super().resizeEvent(event)
        # Eğer bir frame yüklüyse, onu yeniden çiz
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            self.display_frame(self.current_frame)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.avi *.mp4 *.mov)"
        )
        
        if file_path:
            if self.is_live_playing:
                self.is_live_playing = False
                self.live_timer.stop()
                self.live_btn.setText("Live Preview")
            
            self.analysis_results = None
            self.centerline_points = None
            self.prev_live_frame = None
            
            if hasattr(self, 'roi'):
                delattr(self, 'roi')
            
            self.avg_speed_card.update_value("-")
            self.median_speed_card.update_value("-")
            self.std_speed_card.update_value("-")
            self.min_speed_card.update_value("-")
            self.max_speed_card.update_value("-")
            self.valid_label.setText("Valid: -")
            self.alias_label.setText("Aliasing: -")
            
            self.video_name_label.setText("Video: -")
            self.show_uzd_btn.setEnabled(False)
            self.show_histogram_btn.setEnabled(False)
            self.save_csv_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.live_btn.setEnabled(False)
            
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
            
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps == 0 or self.fps > 1000:
                self.fps = 30.0
                self.status_label.setText(f"FPS could not be read, using default {self.fps}")
            else:
                self.status_label.setText(f"Video loaded - FPS: {self.fps:.1f}")
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            video_name = os.path.basename(file_path)
            self.video_name_label.setText(f"Video: {video_name}")
            self.fps_label.setText(f"FPS: {self.fps:.1f}")
            self.frame_count_label.setText(f"Frame Count: {self.total_frames}")
            
            self.current_frame_index = 0
            ret, frame = self.cap.read()
            if ret:
                self.first_frame = frame.copy()
                self.current_frame = frame.copy()
                self.display_frame(frame)
                self.select_roi_btn.setEnabled(True)
                
                self.frame_slider.setMaximum(self.total_frames - 1)
                self.frame_slider.setValue(0)
                self.frame_slider.setEnabled(True)
                self.prev_frame_btn.setEnabled(False)
                self.next_frame_btn.setEnabled(True if self.total_frames > 1 else False)
                self.frame_info_label.setText(f"Frame: 1 / {self.total_frames}")
                self.frame_nav_frame.setVisible(True)
    
    def display_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape

        # Video label'ın mevcut boyutunu al (dinamik)
        label_size = self.video_label.size()
        max_w = label_size.width() - 20  # Padding için 20 piksel boşluk
        max_h = label_size.height() - 20

        # Minimum boyutları garantile
        if max_w < 400:
            max_w = 400
        if max_h < 300:
            max_h = 300

        scale = min(max_w / w, max_h / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(rgb_frame, (new_w, new_h))
        bytes_per_line = ch * new_w

        q_img = QImage(resized.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))
    
    def edit_fps(self):
        if self.fps:
            new_fps, ok = QInputDialog.getDouble(
                self, "Edit FPS", 
                "Enter actual FPS value:",
                self.fps, 1, 10000, 1
            )
            if ok:
                self.fps = new_fps
                self.fps_label.setText(f"FPS: {self.fps:.1f}")
                self.status_label.setText(f"FPS updated: {self.fps:.1f}")
    
    def edit_scale(self):
        new_scale, ok = QInputDialog.getDouble(
            self, "Edit Scale", 
            "Enter spatial scale value (um/pixel):",
            self.pixel_to_um, 0.001, 100, 3
        )
        if ok:
            self.pixel_to_um = new_scale
            self.scale_label.setText(f"Scale: {self.pixel_to_um} um/pixel")
            self.status_label.setText(f"Scale updated: {self.pixel_to_um} um/pixel")
    
    def prev_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.goto_frame(self.current_frame_index)
    
    def next_frame(self):
        if self.current_frame_index < self.total_frames - 1:
            self.current_frame_index += 1
            self.goto_frame(self.current_frame_index)
    
    def on_frame_slider_changed(self, value):
        if value != self.current_frame_index:
            self.current_frame_index = value
            self.goto_frame(value)
    
    def goto_frame(self, frame_index):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            
            if ret:
                self.current_frame = frame.copy()
                self.display_frame(frame)
                
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(frame_index)
                self.frame_slider.blockSignals(False)
                
                self.frame_info_label.setText(f"Frame: {frame_index + 1} / {self.total_frames}")
                
                self.prev_frame_btn.setEnabled(frame_index > 0)
                self.next_frame_btn.setEnabled(frame_index < self.total_frames - 1)
                
                self.status_label.setText(f"Showing frame {frame_index + 1}")
    
    def select_roi(self):
        if self.current_frame is not None:
            self.status_label.setText("Draw region with mouse for ROI selection and press SPACE/ENTER")
            
            roi = cv2.selectROI("Region Selection - Press SPACE/ENTER to confirm, C to cancel", 
                               self.current_frame, False)
            cv2.destroyWindow("Region Selection - Press SPACE/ENTER to confirm, C to cancel")
            
            if roi[2] > 0 and roi[3] > 0:
                self.roi = roi
                x, y, w, h = roi
                
                roi_image = self.current_frame[y:y+h, x:x+w].copy()
                
                method_choice = self._ask_segmentation_method()
                
                if method_choice == "sam":
                    self._segment_with_sam(roi, roi_image)
                elif method_choice == "classic":
                    self._segment_with_classic(roi, roi_image)
                else:
                    self.status_label.setText("Islem iptal edildi")
    
    def _ask_segmentation_method(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Segmentasyon Yontemi")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        info_label = QLabel("Damar tespiti icin yontem secin:")
        info_label.setStyleSheet("font-size: 14px; font-weight: 600; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        sam_btn = ModernButton("SAM (Segment Anything)", primary=True)
        sam_btn.setToolTip("Yapay zeka tabanli - Daha dogru ama yavash (GPU onerilir)")
        
        classic_btn = ModernButton("Klasik Yontem")
        classic_btn.setToolTip("Geleneksel goruntu isleme - Hizli")
        
        cancel_btn = ModernButton("Iptal")
        
        result = {"choice": None}
        
        def on_sam():
            result["choice"] = "sam"
            dialog.accept()
        
        def on_classic():
            result["choice"] = "classic"
            dialog.accept()
        
        def on_cancel():
            result["choice"] = None
            dialog.reject()
        
        sam_btn.clicked.connect(on_sam)
        classic_btn.clicked.connect(on_classic)
        cancel_btn.clicked.connect(on_cancel)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(sam_btn)
        btn_layout.addWidget(classic_btn)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        
        dialog.exec()
        return result["choice"]
    
    def _segment_with_sam(self, roi, roi_image):
        x, y, w, h = roi
        
        self.status_label.setText("SAM modeli yukleniyor...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        QApplication.processEvents()
        
        try:
            sam_segmenter = SAMVesselSegmenter(model_type="vit_b")
            
            self.progress_bar.setValue(20)
            QApplication.processEvents()
            
            if not sam_segmenter.load_model():
                QMessageBox.warning(
                    self, 
                    "SAM Hatasi", 
                    "SAM modeli yuklenemedi.\n\n"
                    "Lutfen su adimlari izleyin:\n"
                    "1. pip install segment-anything torch torchvision\n"
                    "2. sam_vit_b_01ec64.pth dosyasini indirin:\n"
                    "   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
                    "3. Dosyayi proje klasorune kopyalayin"
                )
                self.progress_bar.setVisible(False)
                self._segment_with_classic(roi, roi_image)
                return
            
            self.status_label.setText("SAM ile damar segmentasyonu yapiliyor...")
            self.progress_bar.setValue(40)
            QApplication.processEvents()
            
            self._sam_interactive_segmentation(roi, roi_image, sam_segmenter)
            
        except Exception as e:
            QMessageBox.warning(self, "SAM Hatasi", f"SAM hatasi: {str(e)}\n\nKlasik yonteme geciliyor...")
            self.progress_bar.setVisible(False)
            self._segment_with_classic(roi, roi_image)
    
    def _sam_interactive_segmentation(self, roi, roi_image, sam_segmenter):
        x, y, w, h = roi
        
        display_img = roi_image.copy()
        if len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        
        selected_points = []
        point_labels = []
        current_mask = None
        temp_display = display_img.copy()
        
        def mouse_callback(event, mx, my, flags, param):
            nonlocal temp_display, selected_points, point_labels, current_mask
            
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_points.append([mx, my])
                point_labels.append(1)
                self._update_sam_display(display_img, selected_points, point_labels, current_mask, temp_display)
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                selected_points.append([mx, my])
                point_labels.append(0)
                self._update_sam_display(display_img, selected_points, point_labels, current_mask, temp_display)
        
        cv2.putText(temp_display, "SOL TIK: Damar sec | SAG TIK: Arka plan | S: Segment | ENTER: Onayla", 
                   (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow("SAM Segmentasyon", temp_display)
        cv2.setMouseCallback("SAM Segmentasyon", mouse_callback)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                if len(selected_points) > 0:
                    self.status_label.setText("SAM segmentasyon yapiliyor...")
                    self.progress_bar.setValue(60)
                    QApplication.processEvents()
                    
                    mask, score = sam_segmenter.segment_vessel_with_points(
                        roi_image, selected_points, point_labels
                    )
                    
                    if mask is not None:
                        current_mask = mask
                        self._update_sam_display(display_img, selected_points, point_labels, current_mask, temp_display)
                        self.status_label.setText(f"Segmentasyon tamamlandi - Skor: {score:.2f}")
                    
                    self.progress_bar.setValue(80)
                    QApplication.processEvents()
            
            elif key == 13 or key == ord(' '):
                if current_mask is not None:
                    cv2.destroyWindow("SAM Segmentasyon")
                    
                    self.progress_bar.setValue(90)
                    QApplication.processEvents()
                    
                    centerline_points = sam_segmenter.extract_centerline_from_mask(current_mask)
                    
                    if len(centerline_points) < 10:
                        height, width = roi_image.shape[:2]
                        center_x = width // 2
                        centerline_points = np.array([[row, center_x] for row in range(height)], dtype=np.int32)
                    
                    self.centerline_points = centerline_points
                    self.vessel_mask = current_mask
                    
                    self.progress_bar.setValue(100)
                    QApplication.processEvents()
                    self.progress_bar.setVisible(False)
                    
                    self._show_centerline_confirmation(roi, centerline_points, roi_image)
                    break
                else:
                    cv2.putText(temp_display, "Once 'S' ile segment edin!", 
                               (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.imshow("SAM Segmentasyon", temp_display)
            
            elif key == ord('r') or key == ord('R'):
                selected_points = []
                point_labels = []
                current_mask = None
                temp_display = display_img.copy()
                cv2.putText(temp_display, "SOL TIK: Damar sec | SAG TIK: Arka plan | S: Segment | ENTER: Onayla", 
                           (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imshow("SAM Segmentasyon", temp_display)
            
            elif key == 27 or key == ord('c') or key == ord('C'):
                cv2.destroyWindow("SAM Segmentasyon")
                self.progress_bar.setVisible(False)
                self.status_label.setText("SAM segmentasyon iptal edildi")
                break
    
    def _update_sam_display(self, original, points, labels, mask, temp_display):
        temp_display[:] = original.copy()
        
        if mask is not None:
            mask_colored = np.zeros_like(temp_display)
            mask_colored[:, :, 1] = mask
            temp_display[:] = cv2.addWeighted(temp_display, 0.7, mask_colored, 0.3, 0)
        
        for i, (pt, label) in enumerate(zip(points, labels)):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(temp_display, (int(pt[0]), int(pt[1])), 5, color, -1)
        
        cv2.putText(temp_display, "SOL TIK: Damar sec | SAG TIK: Arka plan | S: Segment | ENTER: Onayla", 
                   (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow("SAM Segmentasyon", temp_display)
    
    def _segment_with_classic(self, roi, roi_image):
        x, y, w, h = roi
        
        self.status_label.setText("Damar merkez cizgisi otomatik tespit ediliyor...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        
        from vessel_processing import VesselProcessor
        vessel_proc = VesselProcessor()
        
        self.progress_bar.setValue(30)
        QApplication.processEvents()
        
        centerline_points, skeleton, vessel_mask = vessel_proc.extract_centerline_from_roi(roi_image)
        
        self.progress_bar.setValue(70)
        QApplication.processEvents()
        
        if len(centerline_points) < 10:
            height, width = roi_image.shape[:2] if len(roi_image.shape) == 2 else roi_image.shape[:2]
            center_x = width // 2
            centerline_points = np.array([[row, center_x] for row in range(height)], dtype=np.int32)
        
        self.centerline_points = centerline_points
        
        self.progress_bar.setValue(100)
        QApplication.processEvents()
        self.progress_bar.setVisible(False)
        
        self._show_centerline_confirmation(roi, centerline_points, roi_image)
    
    def _show_centerline_confirmation(self, roi, centerline_points, roi_image):
        x, y, w, h = roi
        
        display_img = roi_image.copy()
        if len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        
        for i in range(len(centerline_points) - 1):
            pt1_y, pt1_x = centerline_points[i]
            pt2_y, pt2_x = centerline_points[i + 1]
            cv2.line(display_img, (int(pt1_x), int(pt1_y)), (int(pt2_x), int(pt2_y)), (0, 0, 255), 2)
        
        cv2.putText(display_img, "ENTER: Onayla | M: Manuel Ciz | R: Yeniden Otomatik", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Damar Merkez Cizgisi - Onay", display_img)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 13 or key == ord(' '):
                cv2.destroyWindow("Damar Merkez Cizgisi - Onay")
                self._finalize_roi_selection(roi, centerline_points, "Otomatik")
                break
            
            elif key == ord('m') or key == ord('M'):
                cv2.destroyWindow("Damar Merkez Cizgisi - Onay")
                self._manual_centerline_draw(roi, roi_image)
                break
            
            elif key == ord('r') or key == ord('R'):
                cv2.destroyWindow("Damar Merkez Cizgisi - Onay")
                from vessel_processing import VesselProcessor
                vessel_proc = VesselProcessor()
                new_points, _, _ = vessel_proc.extract_centerline_from_roi(roi_image)
                if len(new_points) >= 10:
                    centerline_points = new_points
                self._show_centerline_confirmation(roi, centerline_points, roi_image)
                break
            
            elif key == 27 or key == ord('c') or key == ord('C'):
                cv2.destroyWindow("Damar Merkez Cizgisi - Onay")
                self.status_label.setText("Islem iptal edildi")
                break
    
    def _manual_centerline_draw(self, roi, roi_image):
        x, y, w, h = roi
        
        display_img = roi_image.copy()
        if len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        
        points_selected = []
        temp_display = display_img.copy()
        
        def mouse_callback(event, mx, my, flags, param):
            nonlocal temp_display, points_selected
            
            if event == cv2.EVENT_LBUTTONDOWN:
                points_selected.append((mx, my))
                temp_display = display_img.copy()
                
                for pt in points_selected:
                    cv2.circle(temp_display, pt, 5, (0, 255, 0), -1)
                
                if len(points_selected) >= 2:
                    for i in range(len(points_selected) - 1):
                        cv2.line(temp_display, points_selected[i], points_selected[i+1], (0, 0, 255), 2)
                
                info_text = f"Nokta: {len(points_selected)} | ENTER: Onayla | C: Iptal | R: Sifirla"
                cv2.putText(temp_display, info_text, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.imshow("Manuel Cizim - Noktalar secin", temp_display)
        
        cv2.putText(temp_display, "Damar uzerinde noktalar secin (en az 2)", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("Manuel Cizim - Noktalar secin", temp_display)
        cv2.setMouseCallback("Manuel Cizim - Noktalar secin", mouse_callback)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if (key == 13 or key == ord(' ')) and len(points_selected) >= 2:
                cv2.destroyWindow("Manuel Cizim - Noktalar secin")
                
                centerline_points = self._interpolate_points(points_selected, h)
                self.centerline_points = centerline_points
                self._finalize_roi_selection(roi, centerline_points, "Manuel")
                break
            
            elif key == ord('r') or key == ord('R'):
                points_selected = []
                temp_display = display_img.copy()
                cv2.putText(temp_display, "Damar uzerinde noktalar secin (en az 2)", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.imshow("Manuel Cizim - Noktalar secin", temp_display)
            
            elif key == 27 or key == ord('c') or key == ord('C'):
                cv2.destroyWindow("Manuel Cizim - Noktalar secin")
                self.status_label.setText("Manuel cizim iptal edildi")
                break
    
    def _interpolate_points(self, selected_points, height):
        if len(selected_points) < 2:
            return np.array([[row, selected_points[0][0]] for row in range(height)], dtype=np.int32)
        
        selected_points = sorted(selected_points, key=lambda p: p[1])
        
        all_points = []
        for i in range(len(selected_points) - 1):
            p1 = selected_points[i]
            p2 = selected_points[i + 1]
            
            dist = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
            num_interp = max(dist, 10)
            
            for t in np.linspace(0, 1, num_interp):
                x_interp = int(p1[0] * (1 - t) + p2[0] * t)
                y_interp = int(p1[1] * (1 - t) + p2[1] * t)
                all_points.append([y_interp, x_interp])
        
        return np.array(all_points, dtype=np.int32)
    
    def _finalize_roi_selection(self, roi, centerline_points, method):
        x, y, w, h = roi
        self.centerline_points = centerline_points
        
        frame_with_roi = self.current_frame.copy()
        
        cv2.rectangle(frame_with_roi, 
                    (roi[0], roi[1]), 
                    (roi[0] + roi[2], roi[1] + roi[3]), 
                    (0, 255, 0), 3)
        
        if centerline_points is not None and len(centerline_points) > 0:
            for i in range(len(centerline_points) - 1):
                pt1_y, pt1_x = centerline_points[i]
                pt2_y, pt2_x = centerline_points[i + 1]
                
                pt1 = (x + int(pt1_x), y + int(pt1_y))
                pt2 = (x + int(pt2_x), y + int(pt2_y))
                
                cv2.line(frame_with_roi, pt1, pt2, (255, 0, 0), 2)
            
            self.status_label.setText(f"ROI: {roi[2]}x{roi[3]} px | Cizgi: {len(centerline_points)} nokta ({method})")
        else:
            self.status_label.setText(f"ROI: {roi[2]}x{roi[3]} px | Merkez cizgisi bulunamadi")
        
        self.display_frame(frame_with_roi)
        self.analyze_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
    
    def start_analysis(self):
        if not hasattr(self, 'roi') or self.video_path is None:
            self.status_label.setText("Please load video and select ROI first")
            return
        
        self.analyze_btn.setEnabled(False)
        self.select_roi_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.analysis_thread = AnalysisThread(
            self.video_path,
            self.roi,
            self.fps,
            self.pixel_to_um,
            centerline_points=getattr(self, 'centerline_points', None)
        )
        
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.analysis_finished)
        self.analysis_thread.error.connect(self.analysis_error)
        
        self.analysis_thread.start()
        self.status_label.setText("Analysis started...")
    
    def update_progress(self, current, total, message):
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def analysis_finished(self, results):
        self.analysis_results = results
        self.progress_bar.setVisible(False)
        
        self.avg_speed_card.update_value(f"{results.get('mean', 0):.1f}")
        self.median_speed_card.update_value(f"{results.get('median', 0):.1f}")
        self.std_speed_card.update_value(f"{results.get('std', 0):.1f}")
        self.min_speed_card.update_value(f"{results.get('min', 0):.1f}")
        self.max_speed_card.update_value(f"{results.get('max', 0):.1f}")
        
        self.valid_label.setText(f"Valid: {results.get('n_valid', 0)}")
        self.alias_label.setText(f"Aliasing: {results.get('n_alias', 0)}")
        
        self.show_uzd_btn.setEnabled(True)
        self.show_histogram_btn.setEnabled(True)
        self.save_csv_btn.setEnabled(True)
        
        self.analyze_btn.setEnabled(True)
        self.select_roi_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        self.status_label.setText(
            f"Analysis completed - {results.get('n_valid', 0)} valid measurements | "
            f"Average: {results.get('mean', 0):.1f} um/s"
        )
    
    def toggle_live_preview(self):
        if not self.is_live_playing:
            if not hasattr(self, 'roi') or self.video_path is None:
                self.status_label.setText("Please load video and select ROI first")
                return
            
            tracking_method = self._ask_tracking_method()
            if tracking_method is None:
                return
            
            self.use_sam_tracking = (tracking_method == "sam")
            
            self.is_live_playing = True
            self.live_btn.setText("Stop")
            
            if hasattr(self, 'centerline_points') and self.centerline_points is not None:
                self.live_centerline = self.centerline_points.copy()
                self.original_centerline = self.centerline_points.copy()
            else:
                self.live_centerline = None
                self.original_centerline = None
            
            self.live_frame_count = 0
            self.live_roi = self.roi
            self.original_roi = self.roi
            self.cumulative_displacement = (0, 0)
            self.original_intensity_profile = None
            
            x, y, w, h = self.roi
            if len(self.current_frame.shape) == 3:
                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_frame
            roi_frame_raw = gray[y:y+h, x:x+w].copy()
            
            if self.use_sam_tracking:
                self.status_label.setText("SAM tracker baslatiliyor...")
                QApplication.processEvents()
                
                self.hybrid_tracker = create_vessel_tracker(use_sam=True, sam_interval=5)
                
                initial_mask = getattr(self, 'vessel_mask', None)
                initial_points = None
                
                if initial_mask is None and self.live_centerline is not None:
                    initial_points = [[int(p[1]), int(p[0])] for p in self.live_centerline[::len(self.live_centerline)//5 + 1]]
                
                success = self.hybrid_tracker.initialize(roi_frame_raw, initial_points, initial_mask)
                
                if not success:
                    QMessageBox.warning(self, "SAM Hatasi", "SAM tracker baslatılamadi. Klasik yonteme geciliyor.")
                    self.use_sam_tracking = False
                else:
                    if self.hybrid_tracker.current_centerline is not None:
                        self.live_centerline = self.hybrid_tracker.current_centerline.copy()
                    self.status_label.setText("SAM tracker aktif - Her kare izleniyor")
            
            if not self.use_sam_tracking:
                from vessel_processing import VesselProcessor
                self.vessel_processor = VesselProcessor()
                self.vessel_processor.reset_tracking_state()
            
            roi_frame_blur = cv2.GaussianBlur(roi_frame_raw, (7, 7), 0)
            self.prev_live_frame = cv2.equalizeHist(roi_frame_blur)
            self.prev_live_frame_raw = roi_frame_raw.copy()
            
            interval = int(1000 / self.fps) if self.fps > 0 else 33
            if self.use_sam_tracking:
                interval = max(interval, 100)
            self.live_timer.start(interval)
            
            self.analyze_btn.setEnabled(False)
            self.select_roi_btn.setEnabled(False)
            self.load_btn.setEnabled(False)
            
            if not self.use_sam_tracking:
                self.status_label.setText("Live preview started - Klasik takip aktif")
        else:
            self.is_live_playing = False
            self.live_btn.setText("Live Preview")
            self.live_timer.stop()
            
            if hasattr(self, 'hybrid_tracker') and self.hybrid_tracker is not None:
                self.hybrid_tracker.reset()
            
            if hasattr(self, 'centerline_points') and self.centerline_points is not None:
                self.live_centerline = self.centerline_points.copy()
            
            self.analyze_btn.setEnabled(True)
            self.select_roi_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.status_label.setText("Live preview stopped")
    
    def _ask_tracking_method(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Takip Yontemi")
        dialog.setMinimumWidth(450)
        
        layout = QVBoxLayout()
        
        info_label = QLabel("Live preview icin takip yontemi secin:")
        info_label.setStyleSheet("font-size: 14px; font-weight: 600; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        sam_btn = ModernButton("SAM AI Takip", primary=True)
        sam_btn.setToolTip("Her kare icin SAM ile damar tespiti - Cok dogru ama yavas")
        
        classic_btn = ModernButton("Klasik Takip")
        classic_btn.setToolTip("Optical Flow tabanli - Hizli ama kayabilir")
        
        cancel_btn = ModernButton("Iptal")
        
        result = {"choice": None}
        
        def on_sam():
            result["choice"] = "sam"
            dialog.accept()
        
        def on_classic():
            result["choice"] = "classic"
            dialog.accept()
        
        def on_cancel():
            result["choice"] = None
            dialog.reject()
        
        sam_btn.clicked.connect(on_sam)
        classic_btn.clicked.connect(on_classic)
        cancel_btn.clicked.connect(on_cancel)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(sam_btn)
        btn_layout.addWidget(classic_btn)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        note_label = QLabel("Not: SAM takip her 5 karede tam inference yapar,\naradaki karelerde mask propagation kullanir.")
        note_label.setStyleSheet("font-size: 11px; color: #757575; margin-top: 10px;")
        layout.addWidget(note_label)
        
        dialog.setLayout(layout)
        dialog.exec()
        
        return result["choice"]
    
    def update_live_frame(self):
        if not hasattr(self, 'roi'):
            self.live_timer.stop()
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.live_timer.stop()
                return
            
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            self.live_roi = getattr(self, 'original_roi', self.roi)
            x, y, w, h = self.live_roi
            
            roi_frame = gray[y:y+h, x:x+w].copy()
            roi_frame_blur = cv2.GaussianBlur(roi_frame, (7, 7), 0)
            self.prev_live_frame = cv2.equalizeHist(roi_frame_blur)
            self.prev_live_frame_raw = roi_frame.copy()
            self.prev_full_frame = gray.copy()
            
            if hasattr(self, 'centerline_points') and self.centerline_points is not None:
                self.live_centerline = self.centerline_points.copy()
                
            if getattr(self, 'use_sam_tracking', False) and hasattr(self, 'hybrid_tracker'):
                initial_mask = getattr(self, 'vessel_mask', None)
                initial_points = None
                if initial_mask is None and self.live_centerline is not None:
                    initial_points = [[int(p[1]), int(p[0])] for p in self.live_centerline[::len(self.live_centerline)//5 + 1]]
                self.hybrid_tracker.initialize(roi_frame, initial_points, initial_mask)
            
            self.cumulative_displacement = (0, 0)
            return
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        live_roi = getattr(self, 'live_roi', self.roi)
        x, y, w, h = live_roi
        
        use_sam = getattr(self, 'use_sam_tracking', False)
        
        if use_sam and hasattr(self, 'hybrid_tracker'):
            self._update_live_frame_sam(frame, gray, x, y, w, h)
        else:
            self._update_live_frame_classic(frame, gray, x, y, w, h)
    
    def _update_live_frame_sam(self, frame, gray, x, y, w, h):
        curr_roi_raw = gray[y:y+h, x:x+w].copy()
        
        try:
            mask, centerline, was_full_inference = self.hybrid_tracker.track_frame(curr_roi_raw)
            
            if centerline is not None and len(centerline) > 0:
                self.live_centerline = centerline.copy()
            
            self.live_frame_count = getattr(self, 'live_frame_count', 0) + 1
            
            curr_roi = cv2.GaussianBlur(curr_roi_raw, (7, 7), 0)
            curr_roi = cv2.equalizeHist(curr_roi)
            
            flow = None
            if self.prev_live_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_live_frame, curr_roi, None,
                    pyr_scale=0.5, levels=4, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                global_flow_x = np.median(flow[..., 0])
                global_flow_y = np.median(flow[..., 1])
                flow[..., 0] -= global_flow_x
                flow[..., 1] -= global_flow_y
            
            vis_frame = frame.copy()
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if mask is not None:
                mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
                mask_colored[:, :, 1] = mask
                roi_vis = vis_frame[y:y+h, x:x+w]
                vis_frame[y:y+h, x:x+w] = cv2.addWeighted(roi_vis, 0.7, mask_colored, 0.3, 0)
            
            if self.live_centerline is not None and len(self.live_centerline) > 0:
                for i in range(len(self.live_centerline) - 1):
                    pt1_y, pt1_x = self.live_centerline[i]
                    pt2_y, pt2_x = self.live_centerline[i + 1]
                    pt1 = (x + int(pt1_x), y + int(pt1_y))
                    pt2 = (x + int(pt2_x), y + int(pt2_y))
                    cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 2)
                
                centerline_speeds = []
                if flow is not None:
                    step = max(1, len(self.live_centerline) // 15)
                    for idx in range(0, len(self.live_centerline), step):
                        pt = self.live_centerline[idx]
                        pt_y, pt_x = int(pt[0]), int(pt[1])
                        
                        if 0 <= pt_y < flow.shape[0] and 0 <= pt_x < flow.shape[1]:
                            fx, fy = flow[pt_y, pt_x]
                            speed_pixels = np.sqrt(fx**2 + fy**2)
                            
                            if speed_pixels > 0.3:
                                speed_um = speed_pixels * self.pixel_to_um * self.fps
                                centerline_speeds.append(speed_um)
                                
                                start_point = (x + pt_x, y + pt_y)
                                end_point = (int(x + pt_x + fx * 4), int(y + pt_y + fy * 4))
                                cv2.arrowedLine(vis_frame, start_point, end_point, 
                                              (0, 255, 255), 2, tipLength=0.4)
                
                if centerline_speeds:
                    mean_speed = np.mean(centerline_speeds)
                    cv2.putText(vis_frame, f"Hiz: {mean_speed:.0f} um/s", 
                               (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 255), 2)
            
            inference_text = "SAM" if was_full_inference else "Prop"
            cv2.putText(vis_frame, f"[{inference_text}]", 
                       (x + 5, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
            
            self.display_frame(vis_frame)
            
            current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            status_text = f"SAM Live: Frame {current_frame_num}/{self.total_frames}"
            if was_full_inference:
                status_text += " | SAM inference yapildi"
            self.status_label.setText(status_text)
            
        except Exception as e:
            print(f"SAM live frame hatasi: {e}")
        
        curr_roi = cv2.GaussianBlur(curr_roi_raw, (7, 7), 0)
        self.prev_live_frame = cv2.equalizeHist(curr_roi)
        self.prev_live_frame_raw = curr_roi_raw.copy()
        self.prev_full_frame = gray.copy()
    
    def _update_live_frame_classic(self, frame, gray, x, y, w, h):
        live_roi = getattr(self, 'live_roi', self.roi)
        x, y, w, h = live_roi
        
        if hasattr(self, 'prev_full_frame') and hasattr(self, 'vessel_processor'):
            original_roi = getattr(self, 'original_roi', self.roi)
            centerline = getattr(self, 'original_centerline', None)
            new_roi, roi_displacement = self.vessel_processor.track_roi(
                self.prev_full_frame, gray, live_roi,
                original_roi=original_roi, max_drift=35,
                centerline_points=centerline
            )
            self.live_roi = new_roi
            x, y, w, h = new_roi
        
        curr_roi_raw = gray[y:y+h, x:x+w].copy()
        curr_roi = cv2.GaussianBlur(curr_roi_raw, (7, 7), 0)
        curr_roi = cv2.equalizeHist(curr_roi)
        
        if self.prev_live_frame is not None:
            try:
                if hasattr(self, 'live_centerline') and self.live_centerline is not None and hasattr(self, 'prev_live_frame_raw') and hasattr(self, 'vessel_processor'):
                    self.live_frame_count = getattr(self, 'live_frame_count', 0) + 1
                    cum_disp = getattr(self, 'cumulative_displacement', (0, 0))
                    orig_profile = getattr(self, 'original_intensity_profile', None)
                    
                    original_roi = getattr(self, 'original_roi', self.roi)
                    roi_offset_x = x - original_roi[0]
                    roi_offset_y = y - original_roi[1]
                    
                    self.live_centerline, was_redetected, displacement, new_cum_disp = self.vessel_processor.track_centerline(
                        self.prev_live_frame_raw, curr_roi_raw, self.live_centerline,
                        frame_count=self.live_frame_count,
                        original_centerline=getattr(self, 'original_centerline', None),
                        redetect_interval=30,
                        cumulative_displacement=cum_disp,
                        original_intensity_profile=orig_profile,
                        roi_offset=(roi_offset_y, roi_offset_x)
                    )
                    self.cumulative_displacement = new_cum_disp
                    
                    if was_redetected:
                        self.cumulative_displacement = (0, 0)
                
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_live_frame, curr_roi, None,
                    pyr_scale=0.5, levels=6, winsize=25,
                    iterations=7, poly_n=7, poly_sigma=1.8, flags=0
                )
                
                global_flow_x = np.median(flow[..., 0])
                global_flow_y = np.median(flow[..., 1])
                flow[..., 0] -= global_flow_x
                flow[..., 1] -= global_flow_y
                
                flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (9, 9), 0)
                flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (9, 9), 0)
                
                vis_frame = frame.copy()
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                active_centerline = self.live_centerline if hasattr(self, 'live_centerline') and self.live_centerline is not None else self.centerline_points
                
                if active_centerline is not None and len(active_centerline) > 0:
                    for i in range(len(active_centerline) - 1):
                        pt1_y, pt1_x = active_centerline[i]
                        pt2_y, pt2_x = active_centerline[i + 1]
                        pt1 = (x + int(pt1_x), y + int(pt1_y))
                        pt2 = (x + int(pt2_x), y + int(pt2_y))
                        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 2)
                    
                    centerline_speeds = []
                    
                    step = max(1, len(active_centerline) // 20)
                    for idx in range(0, len(active_centerline), step):
                        pt = active_centerline[idx]
                        pt_y, pt_x = int(pt[0]), int(pt[1])
                        
                        if 0 <= pt_y < flow.shape[0] and 0 <= pt_x < flow.shape[1]:
                            fx, fy = flow[pt_y, pt_x]
                            speed_pixels = np.sqrt(fx**2 + fy**2)
                            
                            if speed_pixels > 0.3:
                                speed_um = speed_pixels * self.pixel_to_um * self.fps
                                centerline_speeds.append(speed_um)
                                
                                start_point = (x + pt_x, y + pt_y)
                                end_point = (int(x + pt_x + fx * 4), int(y + pt_y + fy * 4))
                                cv2.arrowedLine(vis_frame, start_point, end_point, 
                                              (0, 255, 255), 2, tipLength=0.4)
                    
                    if centerline_speeds:
                        mean_speed = np.mean(centerline_speeds)
                        cv2.putText(vis_frame, f"Hiz: {mean_speed:.0f} um/s", 
                                   (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 255), 2)
                
                self.display_frame(vis_frame)
                
                current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                roi_info = f"ROI: ({x},{y})" if hasattr(self, 'live_roi') else ""
                self.status_label.setText(f"Live: Frame {current_frame_num}/{self.total_frames} | {roi_info} | Klasik takip")
            except Exception as e:
                pass
        
        self.prev_live_frame = curr_roi.copy()
        self.prev_live_frame_raw = curr_roi_raw.copy()
        self.prev_full_frame = gray.copy()
    
    def analysis_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.select_roi_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        self.status_label.setText(f"Error: {error_msg}")
        QMessageBox.warning(self, "Analysis Error", error_msg)
    
    def show_uzd(self):
        if self.analysis_results and 'std_image' in self.analysis_results:
            std_image = self.analysis_results['std_image']
            valid_lines = self.analysis_results.get('valid_lines', [])
            
            vis = cv2.cvtColor(std_image, cv2.COLOR_GRAY2BGR)
            for line in valid_lines:
                x1, y1, x2, y2 = line
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            dialog = ImageViewerDialog("Space-Time Diagram (STD)", vis, self)
            dialog.exec()
    
    def show_histogram(self):
        if self.analysis_results and 'all_speeds' in self.analysis_results:
            speeds = self.analysis_results['all_speeds']
            
            if len(speeds) > 0:
                hist_path = self.histogram_generator.create_histogram(speeds)
                
                if hist_path and os.path.exists(hist_path):
                    hist_img = cv2.imread(hist_path)
                    dialog = ImageViewerDialog("Velocity Histogram", hist_img, self)
                    dialog.exec()
                else:
                    QMessageBox.warning(self, "Error", "Histogram could not be created")
            else:
                QMessageBox.information(self, "Information", "No velocity data to display")
    
    def save_csv(self):
        if self.analysis_results:
            video_name = os.path.basename(self.video_path) if self.video_path else ""
            csv_path = self.results_manager.save_to_csv(
                self.analysis_results,
                video_name=video_name
            )
            
            if csv_path:
                self.status_label.setText(f"CSV saved: {csv_path}")
                QMessageBox.information(self, "Success", f"Results saved:\n{csv_path}")
            else:
                QMessageBox.warning(self, "Error", "CSV could not be saved")
    
    def show_about(self):
        dialog = AboutDialog(self)
        dialog.exec()


def main():
    app = QApplication(sys.argv)
    
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
    app.setPalette(palette)
    
    font = QFont("Helvetica", 10)
    app.setFont(font)
    
    window = EritrosidAnalyzer()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
