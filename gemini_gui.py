"""
PyQt6 Frontend for Gemini Sky Removal
Features: Single/Batch processing, prompt editing, real-time results
"""

import sys
import os
from pathlib import Path
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore")

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QFileDialog,
    QListWidget,
    QProgressBar,
    QSplitter,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QMessageBox,
    QStatusBar,
    QMenuBar,
    QMenu,
    QSystemTrayIcon,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QRadioButton,
    QButtonGroup,
    QGridLayout,
    QTreeWidget,
    QTreeWidgetItem,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont, QAction, QPalette, QColor

# Our Gemini sky removal
from sky_removal_methods import SkyRemovalMethods

# Import prompt processor
try:
    from prompt_processor import PromptProcessor
except ImportError:
    print("⚠️  prompt_processor.py not found. Using default prompts.")
    PromptProcessor = None


class FolderScannerThread(QThread):
    """Fast folder scanning thread"""

    files_found = pyqtSignal(list)  # List of image file paths
    scan_finished = pyqtSignal(int)  # Total count

    def __init__(self, folder_path: str):
        super().__init__()
        self.folder_path = folder_path
        self.is_cancelled = False

    def stop(self):
        self.is_cancelled = True

    def run(self):
        """Quickly scan folder and return all image paths"""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        image_files = []

        try:
            for root, dirs, files in os.walk(self.folder_path):
                if self.is_cancelled:
                    break

                for file in files:
                    if self.is_cancelled:
                        break

                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))

            if not self.is_cancelled:
                self.files_found.emit(image_files)
                self.scan_finished.emit(len(image_files))

        except Exception as e:
            print(f"Error scanning folder: {e}")
            self.scan_finished.emit(0)


class SingleThumbnailLoader(QThread):
    """Load thumbnail for single image on demand"""

    thumbnail_ready = pyqtSignal(str, QPixmap)  # path, thumbnail

    def __init__(self, image_path: str, thumb_size: int = 300):
        super().__init__()
        self.image_path = image_path
        self.thumb_size = thumb_size
        self.is_cancelled = False

    def stop(self):
        self.is_cancelled = True

    def run(self):
        """Load single thumbnail"""
        if self.is_cancelled:
            return

        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                # Scale maintaining aspect ratio
                scaled = pixmap.scaled(
                    self.thumb_size,
                    self.thumb_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.thumbnail_ready.emit(self.image_path, scaled)
            else:
                # Create error placeholder
                error_pixmap = QPixmap(self.thumb_size, self.thumb_size)
                error_pixmap.fill(Qt.GlobalColor.lightGray)
                self.thumbnail_ready.emit(self.image_path, error_pixmap)

        except Exception as e:
            print(f"Error loading thumbnail for {self.image_path}: {e}")
            error_pixmap = QPixmap(self.thumb_size, self.thumb_size)
            error_pixmap.fill(Qt.GlobalColor.red)
            self.thumbnail_ready.emit(self.image_path, error_pixmap)


class ProcessingWorker(QThread):
    """Worker thread for processing images with rate limiting"""

    progress_updated = pyqtSignal(int, str)  # progress, message
    image_processed = pyqtSignal(
        str, str, QPixmap
    )  # input_path, output_path, result_pixmap
    finished_all = pyqtSignal(dict)  # results dict
    error_occurred = pyqtSignal(str, str)  # error_message, image_path

    def __init__(
        self,
        sky_remover,
        image_paths: List[str],
        output_dir: str,
        prompt_type: str,
        prompt_kwargs: dict = None,
        rate_limit_delay: float = 6.0,
    ):
        super().__init__()
        self.sky_remover = sky_remover
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.prompt_type = prompt_type
        self.prompt_kwargs = prompt_kwargs or {}
        self.is_cancelled = False
        self.rate_limit_delay = (
            rate_limit_delay  # seconds between requests (default 6s = 10 RPM)
        )
        self.last_request_time = 0
        self.retry_count = 0
        self.max_retries = 3

    def cancel(self):
        self.is_cancelled = True

    def _wait_for_rate_limit(self):
        """Wait to respect rate limits between requests"""
        import time

        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last_request
            self.progress_updated.emit(
                -1,  # Special value to indicate rate limiting
                f"⏱️  Rate limiting: waiting {wait_time:.1f}s before next request...",
            )
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def _handle_rate_limit_error(self, error, image_path):
        """Handle rate limit errors with exponential backoff"""
        import time
        import re

        error_str = str(error).lower()
        if (
            "rate limit" in error_str
            or "quota exceeded" in error_str
            or "429" in error_str
        ):
            self.retry_count += 1
            if self.retry_count <= self.max_retries:
                # Exponential backoff: 30s, 60s, 120s
                backoff_time = 30 * (2 ** (self.retry_count - 1))
                self.progress_updated.emit(
                    -1,
                    f"🚦 Rate limit hit! Retrying in {backoff_time}s (attempt {self.retry_count}/{self.max_retries})...",
                )
                time.sleep(backoff_time)
                return True  # Retry
            else:
                self.error_occurred.emit(
                    f"Rate limit exceeded after {self.max_retries} retries. Please wait before trying again.",
                    image_path,
                )
                return False  # Don't retry
        return False  # Not a rate limit error

    def run(self):
        results = {}
        total = len(self.image_paths)
        import time

        for i, image_path in enumerate(self.image_paths, 1):
            if self.is_cancelled:
                break

            # Reset retry count for each image
            self.retry_count = 0

            while self.retry_count <= self.max_retries:
                try:
                    # Apply rate limiting (skip for first request)
                    if i > 1:
                        self._wait_for_rate_limit()

                    self.progress_updated.emit(
                        int((i - 1) / total * 100),
                        f"Processing {i}/{total}: {os.path.basename(image_path)}",
                    )

                    # Create output path
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = os.path.join(
                        self.output_dir, f"{image_name}_gemini.jpg"
                    )

                    # Process with Gemini using prompt type
                    result = self.sky_remover.gemini_sky_removal(
                        image_path,
                        output_path,
                        prompt_type=self.prompt_type,
                        **self.prompt_kwargs,
                    )

                    # Convert to QPixmap for display
                    height, width, channel = result.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(
                        result.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format.Format_RGB888,
                    )
                    pixmap = QPixmap.fromImage(q_img)

                    self.image_processed.emit(image_path, output_path, pixmap)
                    results[image_path] = {"success": True, "output_path": output_path}
                    break  # Success, exit retry loop

                except Exception as e:
                    if self._handle_rate_limit_error(e, image_path):
                        continue  # Retry after backoff
                    else:
                        # Not a rate limit error or max retries exceeded
                        error_msg = f"Failed to process {os.path.basename(image_path)}: {str(e)}"
                        self.error_occurred.emit(error_msg, image_path)
                        results[image_path] = {"success": False, "error": str(e)}
                        break  # Exit retry loop

        self.progress_updated.emit(100, "Processing complete!")
        self.finished_all.emit(results)


class GeminiSkyRemovalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sky_remover = None
        self.current_worker = None
        self.settings = QSettings("GeminiSkyRemoval", "App")

        # Background loading workers
        self.folder_scanner = None
        self.thumbnail_loader = None
        self.thumbnail_cache = {}  # Cache for loaded thumbnails

        # File tracking
        self.input_files = []  # List of all input image paths
        self.output_files = {}  # Dict mapping input_path -> output_path

        # Initialize prompt processor
        self.prompt_processor = PromptProcessor() if PromptProcessor else None

        self.init_gemini()
        self.init_ui()
        self.load_settings()

    def init_gemini(self):
        """Initialize Gemini API"""
        try:
            self.sky_remover = SkyRemovalMethods()
            self.api_status = True
            self.api_status_text = "✅ Connected"
        except Exception as e:
            QMessageBox.warning(
                self,
                "API Error",
                f"Failed to initialize Gemini API: {e}\n\n"
                "Please check your GOOGLE_API_KEY in .env file.",
            )
            self.api_status = False
            self.api_status_text = "❌ API Not Connected"

        # Set up API status checking timer (check every 30 seconds)
        self.api_check_timer = QTimer()
        self.api_check_timer.timeout.connect(self.check_api_status)
        self.api_check_timer.start(30000)  # 30 seconds

    def check_api_status(self):
        """Check current API status and update UI"""
        if not self.sky_remover:
            return

        try:
            # Quick API connectivity test - list models (lightweight call)
            import google.generativeai as genai

            models = genai.list_models()

            # Check if our model is available
            gemini_models = [m for m in models if "gemini-2.5-flash" in m.name]
            if gemini_models:
                self.api_status = True
                self.api_status_text = "✅ Connected"
                self.api_status_label.setText(self.api_status_text)
                self.api_status_label.setStyleSheet("font-weight: bold; color: green")
            else:
                self.api_status = False
                self.api_status_text = "⚠️ Model Unavailable"
                self.api_status_label.setText(self.api_status_text)
                self.api_status_label.setStyleSheet("font-weight: bold; color: orange")

        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg:
                self.api_status = False
                self.api_status_text = "❌ Quota Exceeded"
                self.api_status_label.setText(self.api_status_text)
                self.api_status_label.setStyleSheet("font-weight: bold; color: red")
            else:
                self.api_status = False
                self.api_status_text = "❌ Connection Error"
                self.api_status_label.setText(self.api_status_text)
                self.api_status_label.setStyleSheet("font-weight: bold; color: red")

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🦍 Gemini Sky Removal - Banana Nano Model")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Results
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([400, 1000])

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - " + self.api_status_text)

        # Menu bar
        self.create_menu_bar()

        # Apply dark theme
        self.apply_dark_theme()

    def create_left_panel(self):
        """Create the left control panel - streamlined"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)

        # Compact Header with API Status
        header_layout = QHBoxLayout()
        title = QLabel("🤖 Gemini")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header_layout.addWidget(title)

        self.api_status_label = QLabel(self.api_status_text)
        self.api_status_label.setStyleSheet(
            "font-size: 11px; color: " + ("green" if self.api_status else "red")
        )
        header_layout.addWidget(self.api_status_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Compact File Selection
        file_group = QGroupBox("Input")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(3)

        # Combined file selection buttons
        btn_layout = QHBoxLayout()
        single_btn = QPushButton("📄 Image")
        single_btn.clicked.connect(self.select_single_file)
        batch_btn = QPushButton("📁 Folder")
        batch_btn.clicked.connect(self.select_batch_folder)
        clear_btn = QPushButton("🗑️")
        clear_btn.clicked.connect(self.clear_files)
        clear_btn.setMaximumWidth(40)

        btn_layout.addWidget(single_btn)
        btn_layout.addWidget(batch_btn)
        btn_layout.addWidget(clear_btn)
        file_layout.addLayout(btn_layout)

        # Compact selection info
        self.selection_info = QLabel("No files selected")
        self.selection_info.setStyleSheet("font-size: 10px; color: gray;")
        file_layout.addWidget(self.selection_info)

        layout.addWidget(file_group)

        # Hidden file list (for compatibility)
        self.file_list = QListWidget()
        self.file_list.setVisible(False)

        # Prompts - Compact
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_layout.setSpacing(3)

        # Preset selector at top
        self.preset_combo = QComboBox()
        # Load available prompts from processor (required)
        if self.prompt_processor:
            try:
                available_prompts = self.prompt_processor.list_available_prompts()
                # Convert to display names
                display_names = {
                    "default": "Default Sky Removal",
                    "conservative": "Conservative Mode",
                    "aggressive": "Aggressive Sky Removal",
                    "building_preservation": "Building Preservation",
                    "urban_scene": "Urban Scene Focus",
                    "custom_template": "Custom Template",
                }
                preset_items = [
                    display_names.get(p, p.title()) for p in available_prompts
                ]
                self.preset_combo.addItems(preset_items)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Preset Error",
                    f"Failed to load presets from JSON: {e}\n\n"
                    "Using basic preset list.",
                )
                self.preset_combo.addItems(["Default Sky Removal"])
        else:
            QMessageBox.critical(
                self,
                "Initialization Error",
                "Prompt processor not available. Cannot load presets without prompts.json.",
            )
            self.preset_combo.addItems(["Error: No Presets"])
        self.preset_combo.currentTextChanged.connect(self.load_preset_prompt)
        prompt_layout.addWidget(self.preset_combo)

        self.prompt_edit = QTextEdit()
        # Initialize with default prompt from processor
        if self.prompt_processor:
            try:
                default_prompt = self.prompt_processor.get_prompt("default")
            except Exception as e:
                default_prompt = "Error: Could not load prompt from JSON file."
        else:
            default_prompt = "Error: Prompt processor not initialized."
        self.prompt_edit.setPlainText(default_prompt)
        self.prompt_edit.setMaximumHeight(120)
        prompt_layout.addWidget(self.prompt_edit)

        layout.addWidget(prompt_group)

        # Processing Options - Compact
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(3)

        self.auto_save = QCheckBox("Auto-save")
        self.auto_save.setChecked(True)

        self.show_preview_checkbox = QCheckBox("Live preview")
        self.show_preview_checkbox.setChecked(True)

        options_layout.addWidget(self.auto_save)
        options_layout.addWidget(self.show_preview_checkbox)

        # API Tier - Compact
        tier_label = QLabel("API Tier:")
        tier_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        options_layout.addWidget(tier_label)

        self.tier_buttons = QButtonGroup()
        self.tiers = {
            "free": {"label": "Free", "delay": 6.0},
            "tier1": {"label": "Tier 1", "delay": 2.0},
            "tier3": {"label": "Tier 3", "delay": 1.0},
        }

        tier_hlayout = QHBoxLayout()
        self.tier_radio_free = QRadioButton(self.tiers["free"]["label"])
        self.tier_radio_t1 = QRadioButton(self.tiers["tier1"]["label"])
        self.tier_radio_t3 = QRadioButton(self.tiers["tier3"]["label"])

        self.tier_buttons.addButton(self.tier_radio_free, 0)
        self.tier_buttons.addButton(self.tier_radio_t1, 1)
        self.tier_buttons.addButton(self.tier_radio_t3, 2)
        self.tier_radio_free.setChecked(True)

        tier_hlayout.addWidget(self.tier_radio_free)
        tier_hlayout.addWidget(self.tier_radio_t1)
        tier_hlayout.addWidget(self.tier_radio_t3)
        options_layout.addLayout(tier_hlayout)

        layout.addWidget(options_group)

        # Process Button
        self.process_btn = QPushButton("🚀 Process with Gemini")
        self.process_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; font-size: 14px; }"
        )
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Cancel button
        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        layout.addWidget(self.cancel_btn)

        # Add stretch to push everything up
        layout.addStretch()

        return panel

    def create_right_panel(self):
        """Create the right panel with unified file tree and preview"""
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        main_layout.setSpacing(5)

        # Compact header
        header = QLabel("Files & Preview")
        header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        main_layout.addWidget(header)

        # Two-column layout: File Tree | Preview
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Left: Unified file tree
        tree_group = QGroupBox("Project Files")
        tree_layout = QVBoxLayout(tree_group)
        tree_layout.setSpacing(3)

        self.unified_tree = QTreeWidget()
        self.unified_tree.setHeaderLabels(["Name", "Type", "Status"])
        self.unified_tree.itemClicked.connect(self.on_tree_item_selected)
        self.unified_tree.setMinimumWidth(300)
        tree_layout.addWidget(self.unified_tree)

        # Statistics
        stats_layout = QHBoxLayout()
        self.input_count_label = QLabel("Input: 0")
        self.output_count_label = QLabel("Output: 0")
        self.input_count_label.setStyleSheet("color: #666; font-size: 10px;")
        self.output_count_label.setStyleSheet("color: #4CAF50; font-size: 10px;")
        stats_layout.addWidget(self.input_count_label)
        stats_layout.addStretch()
        stats_layout.addWidget(self.output_count_label)
        tree_layout.addLayout(stats_layout)

        content_layout.addWidget(tree_group, 1)

        # Right: Preview panel
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(3)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setMinimumHeight(400)

        self.preview_label = QLabel("Select an image to preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("color: gray; font-style: italic;")
        self.preview_scroll.setWidget(self.preview_label)
        preview_layout.addWidget(self.preview_scroll)

        # Preview info
        self.preview_info = QLabel("")
        self.preview_info.setStyleSheet("color: #666; font-size: 10px;")
        preview_layout.addWidget(self.preview_info)

        content_layout.addWidget(preview_group, 2)

        # Keep compatibility trees (hidden)
        self.input_tree = QTreeWidget()
        self.input_tree.setVisible(False)
        self.output_tree = QTreeWidget()
        self.output_tree.setVisible(False)

        # Loading indicator
        self.loading_label = QLabel("Scanning folder...")
        self.loading_label.setVisible(False)
        self.loading_label.setStyleSheet("color: #666; font-style: italic;")
        main_layout.addWidget(self.loading_label)

        return panel

    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_single = QAction("Open Single Image", self)
        open_single.triggered.connect(self.select_single_file)
        file_menu.addAction(open_single)

        open_batch = QAction("Open Batch Folder", self)
        open_batch.triggered.connect(self.select_batch_folder)
        file_menu.addAction(open_batch)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        clear_results = QAction("Clear Results", self)
        clear_results.triggered.connect(self.clear_results)
        tools_menu.addAction(clear_results)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def apply_dark_theme(self):
        """Apply a dark theme to the application"""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

        self.setPalette(palette)

        # Style group boxes
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #555;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #555;
                border-radius: 3px;
            }
        """)

    def select_single_file(self):
        """Select a single image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if file_path:
            self.file_list.clear()
            self.file_list.addItem(file_path)
            self.input_files = [file_path]
            self.update_selection_info()
            self.populate_input_tree()

    def select_batch_folder(self):
        """Select a folder for batch processing"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.file_list.clear()
            self.input_files = []
            if hasattr(self, "unified_tree"):
                self.unified_tree.clear()

            # Show loading
            self.loading_label.setText("Scanning folder for images...")
            self.loading_label.setVisible(True)

            # Start folder scanning
            self.folder_scanner = FolderScannerThread(folder_path)
            self.folder_scanner.files_found.connect(self.on_folder_scanned)
            self.folder_scanner.scan_finished.connect(self.on_folder_scan_complete)
            self.folder_scanner.start()

    def on_folder_scanned(self, image_files: List[str]):
        """Handle scanned image files"""
        self.input_files = sorted(image_files)
        # Add to file list for compatibility with processing
        for img_path in image_files:
            self.file_list.addItem(img_path)
        self.update_selection_info()

    def on_folder_scan_complete(self, total_files: int):
        """Handle completion of folder scanning"""
        self.loading_label.setVisible(False)

        if total_files > 0:
            self.populate_input_tree()
        else:
            QMessageBox.information(
                self, "No Images", "No image files found in the selected folder."
            )

    def populate_input_tree(self):
        """Populate the unified tree with input files"""
        self.unified_tree.clear()

        # Create input folder
        input_root = QTreeWidgetItem(self.unified_tree)
        input_root.setText(0, "📁 Input Files")
        input_root.setText(1, "Folder")
        input_root.setText(2, f"{len(self.input_files)} files")
        input_root.setData(0, Qt.ItemDataRole.UserRole, "input_root")

        # Group files by directory
        dir_items = {}

        for file_path in self.input_files:
            dir_name = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)

            # Create directory item if not exists
            if dir_name not in dir_items:
                dir_item = QTreeWidgetItem(input_root)
                dir_item.setText(0, os.path.basename(dir_name) or "Root")
                dir_item.setText(1, "Directory")
                dir_item.setText(2, "")
                dir_item.setData(0, Qt.ItemDataRole.UserRole, "directory")
                dir_items[dir_name] = dir_item

            # Add file item
            file_item = QTreeWidgetItem(dir_items[dir_name])
            file_item.setText(0, file_name)
            file_item.setText(1, "Image")
            file_item.setText(2, "Ready")
            file_item.setData(0, Qt.ItemDataRole.UserRole, ("input", file_path))

        # Create output folder (initially empty)
        output_root = QTreeWidgetItem(self.unified_tree)
        output_root.setText(0, "✨ Output Files")
        output_root.setText(1, "Folder")
        output_root.setText(2, "0 files")
        output_root.setData(0, Qt.ItemDataRole.UserRole, "output_root")
        self.output_root = output_root

        # Expand input folder
        input_root.setExpanded(True)
        self.input_count_label.setText(f"Input: {len(self.input_files)}")

    def on_tree_item_selected(self, item, column):
        """Handle tree item selection"""
        data = item.data(0, Qt.ItemDataRole.UserRole)

        if isinstance(data, tuple):
            file_type, file_path = data
            if os.path.exists(file_path):
                self.load_preview(file_path)
        elif isinstance(data, str) and data not in [
            "input_root",
            "output_root",
            "directory",
            "errors",
        ]:
            if os.path.exists(data):
                self.load_preview(data)

    def on_input_file_selected(self, item, column):
        """Compatibility method - redirects to unified tree handler"""
        self.on_tree_item_selected(item, column)

    def on_output_file_selected(self, item, column):
        """Compatibility method - redirects to unified tree handler"""
        self.on_tree_item_selected(item, column)

    def load_preview(self, file_path: str):
        """Load preview for selected file"""
        if file_path in self.thumbnail_cache:
            # Use cached thumbnail
            self.display_preview(file_path, self.thumbnail_cache[file_path])
        else:
            # Load thumbnail in background
            self.preview_label.setText("Loading preview...")

            # Stop any existing loader
            if self.thumbnail_loader:
                self.thumbnail_loader.stop()
                self.thumbnail_loader.wait()

            self.thumbnail_loader = SingleThumbnailLoader(file_path)
            self.thumbnail_loader.thumbnail_ready.connect(self.on_preview_ready)
            self.thumbnail_loader.start()

    def on_preview_ready(self, file_path: str, pixmap: QPixmap):
        """Handle preview thumbnail ready"""
        self.thumbnail_cache[file_path] = pixmap
        self.display_preview(file_path, pixmap)

    def display_preview(self, file_path: str, pixmap: QPixmap):
        """Show preview in preview panel"""
        self.preview_label.setPixmap(pixmap)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Update info
        file_name = os.path.basename(file_path)
        try:
            size = os.path.getsize(file_path)
            if size < 1024 * 1024:
                size_text = f"{size / 1024:.1f} KB"
            else:
                size_text = f"{size / (1024 * 1024):.1f} MB"

            self.preview_info.setText(
                f"{file_name} • {pixmap.width()}x{pixmap.height()} • {size_text}"
            )
        except:
            self.preview_info.setText(file_name)

    def clear_files(self):
        """Clear all selected files"""
        # Stop any ongoing loading
        if self.thumbnail_loader:
            self.thumbnail_loader.stop()
        if self.folder_scanner:
            self.folder_scanner.stop()

        self.file_list.clear()
        self.input_files = []
        self.output_files = {}
        self.thumbnail_cache.clear()

        # Clear unified tree
        if hasattr(self, "unified_tree"):
            self.unified_tree.clear()

        # Reset preview
        self.preview_label.setText("Select an image to preview")
        self.preview_label.setPixmap(QPixmap())
        self.preview_info.setText("")

        # Update counts
        self.input_count_label.setText("Input: 0")
        self.output_count_label.setText("Output: 0")

        self.update_selection_info()
        self.loading_label.setVisible(False)

    def refresh_dataset_gallery(self):
        """Refresh the dataset view - now handled by tree view"""
        self.populate_input_tree()

    def add_result_to_tree(self, input_path: str, output_path: str):
        """Add processed result to unified tree"""
        self.output_files[input_path] = output_path

        # Add to output section of unified tree
        if hasattr(self, "output_root"):
            # Find or create output directory
            output_dir = os.path.dirname(output_path)
            dir_item = None

            for i in range(self.output_root.childCount()):
                child = self.output_root.child(i)
                if child.data(0, Qt.ItemDataRole.UserRole) == output_dir:
                    dir_item = child
                    break

            if not dir_item:
                dir_item = QTreeWidgetItem(self.output_root)
                dir_item.setText(0, os.path.basename(output_dir) or "Output")
                dir_item.setText(1, "Directory")
                dir_item.setText(2, "")
                dir_item.setData(0, Qt.ItemDataRole.UserRole, output_dir)

            # Add file item
            file_item = QTreeWidgetItem(dir_item)
            file_item.setText(0, os.path.basename(output_path))
            file_item.setText(1, "Processed")
            file_item.setText(2, "✅")
            file_item.setData(0, Qt.ItemDataRole.UserRole, ("output", output_path))

            # Update counts
            self.output_root.setText(2, f"{len(self.output_files)} files")
            self.output_root.setExpanded(True)
            dir_item.setExpanded(True)

            processed_count = len(self.output_files)
            self.output_count_label.setText(f"Output: {processed_count}")

    def update_selection_info(self):
        """Update the selection info label"""
        count = self.file_list.count()
        if count == 0:
            self.selection_info.setText("No files selected")
        elif count == 1:
            input_filename = os.path.splitext(
                os.path.basename(self.file_list.item(0).text())
            )[0]
            self.selection_info.setText(
                f"1 file selected - output will be saved to '{input_filename}_gemini_results' folder"
            )
        else:
            self.selection_info.setText(
                f"{count} files selected - batch processing with auto-generated output folder"
            )

    def load_preset_prompt(self, preset_name):
        """Load a preset prompt using the prompt processor (JSON only)"""
        if not self.prompt_processor:
            QMessageBox.warning(
                self, "Error", "Prompt processor not available. Cannot load presets."
            )
            return

        # Map display names back to prompt types
        name_to_type = {
            "Default Sky Removal": "default",
            "Conservative Mode": "conservative",
            "Aggressive Sky Removal": "aggressive",
            "Building Preservation": "building_preservation",
            "Urban Scene Focus": "urban_scene",
            "Custom Template": "custom_template",
        }

        prompt_type = name_to_type.get(preset_name, "default")
        try:
            prompt = self.prompt_processor.get_prompt(prompt_type)
            self.prompt_edit.setPlainText(prompt)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Preset Error",
                f"Failed to load preset '{preset_name}' from JSON: {e}\n\n"
                "Please check your prompts.json file.",
            )

    def start_processing(self):
        """Start the processing"""
        if not self.api_status:
            QMessageBox.warning(
                self,
                "API Error",
                "Gemini API is not connected. Please check your API key.",
            )
            return

        # Get selected files
        image_paths = []
        for i in range(self.file_list.count()):
            image_paths.append(self.file_list.item(i).text())

        if not image_paths:
            QMessageBox.warning(
                self, "No Files", "Please select image files to process."
            )
            return

        # Generate output directory automatically
        if self.auto_save.isChecked():
            if len(image_paths) == 1:
                # Single file: create output dir next to input file
                input_dir = os.path.dirname(image_paths[0])
                input_filename = os.path.splitext(os.path.basename(image_paths[0]))[0]
                output_dir = os.path.join(input_dir, f"{input_filename}_gemini_results")
            else:
                # Batch processing: create output dir next to input folder
                # Find common parent directory of all files
                input_dirs = set(os.path.dirname(path) for path in image_paths)
                if len(input_dirs) == 1:
                    # All files in same directory
                    input_dir = list(input_dirs)[0]
                    output_dir = os.path.join(
                        os.path.dirname(input_dir),
                        f"{os.path.basename(input_dir)}_gemini_results",
                    )
                else:
                    # Files in different directories - use parent of first file
                    first_input_dir = os.path.dirname(image_paths[0])
                    output_dir = os.path.join(
                        os.path.dirname(first_input_dir), "gemini_results"
                    )

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Auto-created output directory: {output_dir}")
            self.status_bar.showMessage(
                f"Output directory created: {os.path.basename(output_dir)}"
            )

        else:
            output_dir = None

        # Get prompt type from current preset selection
        current_preset = self.preset_combo.currentText()
        preset_to_type = {
            "Default Sky Removal": "default",
            "Conservative Mode": "conservative",
            "Aggressive Sky Removal": "aggressive",
            "Building Preservation": "building_preservation",
            "Urban Scene Focus": "urban_scene",
            "Custom Template": "custom_template",
        }
        prompt_type = preset_to_type.get(current_preset, "default")

        # Get custom prompt if user edited it
        current_prompt_text = self.prompt_edit.toPlainText().strip()
        default_prompt = ""
        if self.prompt_processor:
            try:
                default_prompt = self.prompt_processor.get_prompt(prompt_type)
            except:
                default_prompt = ""

        # Prepare prompt kwargs
        prompt_kwargs = {}

        # If user modified the prompt, use custom mode
        if current_prompt_text != default_prompt and current_prompt_text:
            prompt_type = "custom"
            prompt_kwargs["custom_prompt"] = current_prompt_text

        # Update UI
        self.process_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Clear previous results
        self.clear_results()

        # Start processing thread
        # Map selected tier to delay
        if self.tier_radio_t3.isChecked():
            rate_limit_delay = self.tiers["tier3"]["delay"]
        elif self.tier_radio_t1.isChecked():
            rate_limit_delay = self.tiers["tier1"]["delay"]
        else:
            rate_limit_delay = self.tiers["free"]["delay"]
        self.current_worker = ProcessingWorker(
            self.sky_remover,
            image_paths,
            output_dir or "",
            prompt_type,
            prompt_kwargs,
            float(rate_limit_delay),
        )
        self.current_worker.progress_updated.connect(self.update_progress)
        self.current_worker.image_processed.connect(self.add_result)
        self.current_worker.error_occurred.connect(self.handle_error)
        self.current_worker.finished_all.connect(self.processing_finished)
        self.current_worker.start()

    def cancel_processing(self):
        """Cancel the current processing"""
        if self.current_worker:
            self.current_worker.cancel()
            self.status_bar.showMessage("Cancelling...")

    def update_progress(self, value, message):
        """Update progress bar and status"""
        if value == -1:
            # Special value for rate limiting - don't update progress bar
            self.status_bar.showMessage(message)
        else:
            self.progress_bar.setValue(value)
            self.status_bar.showMessage(message)

    def add_result(self, input_path, output_path, pixmap):
        """Add a processed image result to the UI"""
        # Add to output tree
        self.add_result_to_tree(input_path, output_path)

        # Cache the result pixmap for preview
        if pixmap and not pixmap.isNull():
            self.thumbnail_cache[output_path] = pixmap

    def handle_error(self, error_msg, image_path):
        # Handle processing errors"""
        # Add error entry to unified tree if available
        if hasattr(self, "output_root"):
            # Find or create error folder
            error_dir = None
            for i in range(self.output_root.childCount()):
                child = self.output_root.child(i)
                if child.text(0) == "❌ Errors":
                    error_dir = child
                    break

            if not error_dir:
                error_dir = QTreeWidgetItem(self.output_root)
                error_dir.setText(0, "❌ Errors")
                error_dir.setText(1, "Folder")
                error_dir.setText(2, "")
                error_dir.setData(0, Qt.ItemDataRole.UserRole, "errors")

            # Add error file
            file_item = QTreeWidgetItem(error_dir)
            file_item.setText(0, os.path.basename(image_path))
            file_item.setText(1, "Failed")
            file_item.setText(2, "❌")
            file_item.setData(0, Qt.ItemDataRole.UserRole, ("error", image_path))

            error_dir.setExpanded(True)
            self.output_root.setExpanded(True)

    def processing_finished(self, results):
        """Handle processing completion"""
        self.process_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)

        success_count = sum(1 for r in results.values() if r.get("success", False))
        total_count = len(results)

        self.status_bar.showMessage(
            f"Processing complete! {success_count}/{total_count} successful"
        )

        QMessageBox.information(
            self,
            "Processing Complete",
            f"Successfully processed {success_count} out of {total_count} images.",
        )

        self.current_worker = None

    def clear_results(self):
        """Clear all results from the UI"""
        if hasattr(self, "unified_tree"):
            # Clear only output items
            if hasattr(self, "output_root"):
                while self.output_root.childCount() > 0:
                    self.output_root.takeChild(0)
                self.output_root.setText(2, "0 files")
        self.output_files = {}
        self.output_count_label.setText("Output: 0")

        # Clear output thumbnails from cache
        output_paths = list(self.output_files.values())
        for path in output_paths:
            if path in self.thumbnail_cache:
                del self.thumbnail_cache[path]

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Gemini Sky Removal",
            "🦍 Gemini Sky Removal - Banana Nano Model\n\n"
            "A PyQt6 frontend for Google's Gemini 2.5 Flash API\n"
            "for advanced sky and cloud removal from images.\n\n"
            "Features:\n"
            "• Single and batch image processing\n"
            "• Customizable prompts\n"
            "• Real-time preview\n"
            "• Professional UI\n\n"
            "Built with ❤️ using Gemini AI",
        )

    def load_settings(self):
        """Load application settings"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Load saved prompt, but clear old hardcoded prompts
        saved_prompt = self.settings.value("prompt")
        if saved_prompt:
            # Check if this is an old hardcoded prompt that should be cleared
            old_hardcoded_start = "Please create a segmentation mask"
            if saved_prompt.startswith(old_hardcoded_start):
                # Clear the old saved prompt so JSON prompt is used
                self.settings.remove("prompt")
                print("🧹 Cleared old hardcoded prompt from settings")
            else:
                # Use the saved custom prompt
                self.prompt_edit.setPlainText(saved_prompt)

        # Load selected API tier
        saved_tier = self.settings.value("api_tier", "free")
        if saved_tier == "tier3":
            self.tier_radio_t3.setChecked(True)
        elif saved_tier == "tier1":
            self.tier_radio_t1.setChecked(True)
        else:
            self.tier_radio_free.setChecked(True)

    def save_settings(self):
        """Save application settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("prompt", self.prompt_edit.toPlainText())
        # Save selected API tier
        if self.tier_radio_t3.isChecked():
            self.settings.setValue("api_tier", "tier3")
        elif self.tier_radio_t1.isChecked():
            self.settings.setValue("api_tier", "tier1")
        else:
            self.settings.setValue("api_tier", "free")

    def closeEvent(self, event):
        """Handle application close"""
        self.save_settings()

        # Stop all workers
        if self.current_worker:
            self.current_worker.cancel()
            self.current_worker.wait()
        if self.thumbnail_loader:
            self.thumbnail_loader.stop()
            self.thumbnail_loader.wait()
        if self.folder_scanner:
            self.folder_scanner.stop()
            self.folder_scanner.wait()

        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Gemini Sky Removal")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Banana Nano")

    # Create and show main window
    window = GeminiSkyRemovalGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
