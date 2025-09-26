import os
import sys
from typing import List, Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QFileDialog, QListWidget, QProgressBar, QSplitter, QGroupBox,
    QCheckBox, QComboBox, QMessageBox, QScrollArea, QRadioButton, QButtonGroup,
    QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QPixmap, QFont, QAction, QPalette, QColor

from .workers import FolderScannerThread, SingleThumbnailLoader, ProcessingWorker
from sky_removal_methods import SkyRemovalMethods

try:
    from prompt_processor import PromptProcessor
except ImportError:
    print("⚠️  prompt_processor.py not found. Using default prompts.")
    PromptProcessor = None

class GeminiSkyRemovalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sky_remover = None
        self.current_worker = None
        self.settings = QSettings("GeminiSkyRemoval", "App")
        self.folder_scanner = None
        self.thumbnail_loader = None
        self.thumbnail_cache = {}
        self.pre_existing_results = {}
        self.failed_inputs = {}
        self.input_files: List[str] = []
        self.output_files = {}
        self.prompt_processor = PromptProcessor() if PromptProcessor else None
        self.init_gemini()
        self.init_ui()
        self.load_settings()

    # --- init & status ---
    def init_gemini(self):
        try:
            self.sky_remover = SkyRemovalMethods()
            self.api_status = True
            self.api_status_text = "✅ Connected"
        except Exception as e:
            QMessageBox.warning(self, "API Error", f"Failed to initialize Gemini API: {e}\n\nPlease check your GOOGLE_API_KEY in .env file.")
            self.api_status = False
            self.api_status_text = "❌ API Not Connected"
        self.api_check_timer = QTimer()
        self.api_check_timer.timeout.connect(self.check_api_status)
        self.api_check_timer.start(30000)

    def check_api_status(self):
        if not self.sky_remover:
            return
        try:
            import google.generativeai as genai
            models = genai.list_models()
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
            else:
                self.api_status = False
                self.api_status_text = "❌ Connection Error"
            self.api_status_label.setText(self.api_status_text)
            self.api_status_label.setStyleSheet("font-weight: bold; color: red")

    # --- UI ---
    def init_ui(self):
        self.setWindowTitle("🦍 Gemini Sky Removal - Banana Nano Model")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - " + self.api_status_text)
        self.create_menu_bar()
        self.apply_dark_theme()

    def create_left_panel(self):
        panel = QWidget(); layout = QVBoxLayout(panel); layout.setSpacing(5)
        header_layout = QHBoxLayout(); title = QLabel("🤖 Gemini"); title.setFont(QFont("Arial", 12, QFont.Weight.Bold)); header_layout.addWidget(title)
        self.api_status_label = QLabel(self.api_status_text)
        self.api_status_label.setStyleSheet("font-size: 11px; color: " + ("green" if self.api_status else "red"))
        header_layout.addWidget(self.api_status_label); header_layout.addStretch(); layout.addLayout(header_layout)
        file_group = QGroupBox("Input"); file_layout = QVBoxLayout(file_group); file_layout.setSpacing(3)
        btn_layout = QHBoxLayout(); single_btn = QPushButton("📄 Image"); single_btn.clicked.connect(self.select_single_file); batch_btn = QPushButton("📁 Folder"); batch_btn.clicked.connect(self.select_batch_folder); clear_btn = QPushButton("🗑️"); clear_btn.clicked.connect(self.clear_files); clear_btn.setMaximumWidth(40)
        btn_layout.addWidget(single_btn); btn_layout.addWidget(batch_btn); btn_layout.addWidget(clear_btn); file_layout.addLayout(btn_layout)
        self.selection_info = QLabel("No files selected"); self.selection_info.setStyleSheet("font-size: 10px; color: gray;"); file_layout.addWidget(self.selection_info)
        layout.addWidget(file_group)
        self.file_list = QListWidget(); self.file_list.setVisible(False)
        prompt_group = QGroupBox("Prompt"); prompt_layout = QVBoxLayout(prompt_group); prompt_layout.setSpacing(3)
        self.preset_combo = QComboBox()
        if self.prompt_processor:
            try:
                available_prompts = self.prompt_processor.list_available_prompts()
                display_names = {"default": "Default Sky Removal","conservative": "Conservative Mode","aggressive": "Aggressive Sky Removal","building_preservation": "Building Preservation","urban_scene": "Urban Scene Focus","custom_template": "Custom Template",}
                preset_items = [display_names.get(p, p.title()) for p in available_prompts]
                self.preset_combo.addItems(preset_items)
            except Exception as e:
                QMessageBox.warning(self, "Preset Error", f"Failed to load presets from JSON: {e}\n\nUsing basic preset list.")
                self.preset_combo.addItems(["Default Sky Removal"])
        else:
            QMessageBox.critical(self, "Initialization Error", "Prompt processor not available. Cannot load presets without prompts.json.")
            self.preset_combo.addItems(["Error: No Presets"])
        self.preset_combo.currentTextChanged.connect(self.load_preset_prompt)
        prompt_layout.addWidget(self.preset_combo)
        self.prompt_edit = QTextEdit()
        if self.prompt_processor:
            try:
                default_prompt = self.prompt_processor.get_prompt("default")
            except Exception:
                default_prompt = "Error: Could not load prompt from JSON file."
        else:
            default_prompt = "Error: Prompt processor not initialized."
        self.prompt_edit.setPlainText(default_prompt); self.prompt_edit.setMaximumHeight(120); prompt_layout.addWidget(self.prompt_edit)
        layout.addWidget(prompt_group)
        options_group = QGroupBox("Options"); options_layout = QVBoxLayout(options_group); options_layout.setSpacing(3)
        self.auto_save = QCheckBox("Auto-save"); self.auto_save.setChecked(True)
        self.show_preview_checkbox = QCheckBox("Live preview"); self.show_preview_checkbox.setChecked(True)
        options_layout.addWidget(self.auto_save); options_layout.addWidget(self.show_preview_checkbox)
        tier_label = QLabel("API Tier:"); tier_label.setStyleSheet("font-weight: bold; margin-top: 5px;"); options_layout.addWidget(tier_label)
        self.tier_buttons = QButtonGroup(); self.tiers = {"free": {"label": "Free", "delay": 6.0}, "tier1": {"label": "Tier 1", "delay": 2.0}, "tier3": {"label": "Tier 3", "delay": 1.0}}
        tier_hlayout = QHBoxLayout(); self.tier_radio_free = QRadioButton(self.tiers["free"]["label"]); self.tier_radio_t1 = QRadioButton(self.tiers["tier1"]["label"]); self.tier_radio_t3 = QRadioButton(self.tiers["tier3"]["label"])
        self.tier_buttons.addButton(self.tier_radio_free, 0); self.tier_buttons.addButton(self.tier_radio_t1, 1); self.tier_buttons.addButton(self.tier_radio_t3, 2); self.tier_radio_free.setChecked(True)
        tier_hlayout.addWidget(self.tier_radio_free); tier_hlayout.addWidget(self.tier_radio_t1); tier_hlayout.addWidget(self.tier_radio_t3); options_layout.addLayout(tier_hlayout)
        layout.addWidget(options_group)
        self.process_btn = QPushButton("🚀 Process with Gemini"); self.process_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; font-size: 14px; }"); self.process_btn.clicked.connect(self.start_processing); layout.addWidget(self.process_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); layout.addWidget(self.progress_bar)
        self.cancel_btn = QPushButton("❌ Cancel"); self.cancel_btn.setVisible(False); self.cancel_btn.clicked.connect(self.cancel_processing); layout.addWidget(self.cancel_btn)
        self.retry_failed_btn = QPushButton("🔁 Retry Failed"); self.retry_failed_btn.setEnabled(False); self.retry_failed_btn.setToolTip("No failed images to retry."); self.retry_failed_btn.clicked.connect(self.retry_failed_images); layout.addWidget(self.retry_failed_btn)
        layout.addStretch()
        return panel

    def create_right_panel(self):
        panel = QWidget(); main_layout = QVBoxLayout(panel); main_layout.setSpacing(5)
        header = QLabel("Files & Preview"); header.setFont(QFont("Arial", 11, QFont.Weight.Bold)); main_layout.addWidget(header)
        content_layout = QHBoxLayout(); main_layout.addLayout(content_layout)
        tree_group = QGroupBox("Project Files"); tree_layout = QVBoxLayout(tree_group); tree_layout.setSpacing(3)
        self.unified_tree = QTreeWidget(); self.unified_tree.setHeaderLabels(["Name", "Type", "Status"]); self.unified_tree.itemClicked.connect(self.on_tree_item_selected); self.unified_tree.setMinimumWidth(300); tree_layout.addWidget(self.unified_tree)
        stats_layout = QHBoxLayout(); self.input_count_label = QLabel("Input: 0"); self.output_count_label = QLabel("Output: 0"); self.input_count_label.setStyleSheet("color: #666; font-size: 10px;"); self.output_count_label.setStyleSheet("color: #4CAF50; font-size: 10px;"); stats_layout.addWidget(self.input_count_label); stats_layout.addStretch(); stats_layout.addWidget(self.output_count_label); tree_layout.addLayout(stats_layout)
        content_layout.addWidget(tree_group, 1)
        preview_group = QGroupBox("Preview"); preview_layout = QVBoxLayout(preview_group); preview_layout.setSpacing(3)
        self.preview_scroll = QScrollArea(); self.preview_scroll.setWidgetResizable(True); self.preview_scroll.setMinimumHeight(400)
        self.preview_label = QLabel("Select an image to preview"); self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.preview_label.setStyleSheet("color: gray; font-style: italic;"); self.preview_scroll.setWidget(self.preview_label); preview_layout.addWidget(self.preview_scroll)
        self.preview_info = QLabel(""); self.preview_info.setStyleSheet("color: #666; font-size: 10px;"); preview_layout.addWidget(self.preview_info)
        content_layout.addWidget(preview_group, 2)
        self.input_tree = QTreeWidget(); self.input_tree.setVisible(False)
        self.output_tree = QTreeWidget(); self.output_tree.setVisible(False)
        self.loading_label = QLabel("Scanning folder..."); self.loading_label.setVisible(False); self.loading_label.setStyleSheet("color: #666; font-style: italic;"); main_layout.addWidget(self.loading_label)
        return panel

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_single = QAction("Open Single Image", self); open_single.triggered.connect(self.select_single_file); file_menu.addAction(open_single)
        open_batch = QAction("Open Batch Folder", self); open_batch.triggered.connect(self.select_batch_folder); file_menu.addAction(open_batch)
        file_menu.addSeparator(); exit_action = QAction("Exit", self); exit_action.triggered.connect(self.close); file_menu.addAction(exit_action)
        tools_menu = menubar.addMenu("Tools")
        clear_results = QAction("Clear Results", self); clear_results.triggered.connect(self.clear_results); tools_menu.addAction(clear_results)
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self); about_action.triggered.connect(self.show_about); help_menu.addAction(about_action)

    def apply_dark_theme(self):
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
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 2px solid #4CAF50; border-radius: 5px; margin-top: 1ex; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 10px 0 10px; }
            QListWidget { background-color: #2a2a2a; border: 1px solid #555; border-radius: 3px; }
            QTextEdit { background-color: #2a2a2a; border: 1px solid #555; border-radius: 3px; }
        """)

    # ---- File selection & preview ----
    def select_single_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if file_path:
            self.file_list.clear(); self.file_list.addItem(file_path)
            self.input_files = [file_path]; self.pre_existing_results = {}; self.failed_inputs = {}
            self.update_retry_button_state(); self.update_selection_info(); self.populate_input_tree()

    def select_batch_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.file_list.clear(); self.input_files = []; self.pre_existing_results = {}; self.failed_inputs = {}
            self.update_retry_button_state();
            if hasattr(self, "unified_tree"): self.unified_tree.clear()
            self.loading_label.setText("Scanning folder for images..."); self.loading_label.setVisible(True)
            self.folder_scanner = FolderScannerThread(folder_path)
            self.folder_scanner.files_found.connect(self.on_folder_scanned)
            self.folder_scanner.scan_finished.connect(self.on_folder_scan_complete)
            self.folder_scanner.start()

    def on_folder_scanned(self, image_files: List[str]):
        processed_map = {}
        if self.folder_scanner and getattr(self.folder_scanner, "already_processed", None):
            processed_map = {path: result for path, result in self.folder_scanner.already_processed if path and result and os.path.exists(result)}
        self.pre_existing_results = processed_map
        pending_files = sorted(path for path in image_files if path not in processed_map)
        self.input_files = pending_files
        for img_path in pending_files:
            self.file_list.addItem(img_path)
        skipped_count = len(processed_map)
        if skipped_count:
            message = f"{skipped_count} image{'s' if skipped_count != 1 else ''} already processed and available in results."
            self.status_bar.showMessage(message)
        else:
            self.status_bar.clearMessage()
        self.update_selection_info(); self.update_retry_button_state()

    def on_folder_scan_complete(self, total_files: int):
        self.loading_label.setVisible(False)
        if total_files > 0:
            self.populate_input_tree()
        else:
            QMessageBox.information(self, "No Images", "No image files found in the selected folder.")

    def populate_input_tree(self):
        self.unified_tree.clear()
        input_root = QTreeWidgetItem(self.unified_tree)
        input_root.setText(0, "📁 Input Files"); input_root.setText(1, "Folder"); input_root.setText(2, f"{len(self.input_files)} files"); input_root.setData(0, Qt.ItemDataRole.UserRole, "input_root")
        dir_items = {}
        for file_path in self.input_files:
            dir_name = os.path.dirname(file_path); file_name = os.path.basename(file_path)
            if dir_name not in dir_items:
                dir_item = QTreeWidgetItem(input_root); dir_item.setText(0, os.path.basename(dir_name) or "Root"); dir_item.setText(1, "Directory"); dir_item.setText(2, ""); dir_item.setData(0, Qt.ItemDataRole.UserRole, "directory"); dir_items[dir_name] = dir_item
            file_item = QTreeWidgetItem(dir_items[dir_name]); file_item.setText(0, file_name); file_item.setText(1, "Image"); file_item.setText(2, "Ready"); file_item.setData(0, Qt.ItemDataRole.UserRole, ("input", file_path))
        output_root = QTreeWidgetItem(self.unified_tree); output_root.setText(0, "✨ Output Files"); output_root.setText(1, "Folder"); output_root.setText(2, "0 files"); output_root.setData(0, Qt.ItemDataRole.UserRole, "output_root"); self.output_root = output_root
        self.restore_preexisting_results()
        input_root.setExpanded(True); self.input_count_label.setText(f"Input: {len(self.input_files)}")

    def on_tree_item_selected(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, tuple):
            file_type, file_path = data
            if os.path.exists(file_path):
                self.load_preview(file_path)
        elif isinstance(data, str) and data not in ["input_root", "output_root", "directory", "errors"]:
            if os.path.exists(data):
                self.load_preview(data)

    def load_preview(self, file_path: str):
        if file_path in self.thumbnail_cache:
            self.display_preview(file_path, self.thumbnail_cache[file_path])
        else:
            self.preview_label.setText("Loading preview...")
            if self.thumbnail_loader:
                self.thumbnail_loader.stop(); self.thumbnail_loader.wait()
            self.thumbnail_loader = SingleThumbnailLoader(file_path)
            self.thumbnail_loader.thumbnail_ready.connect(self.on_preview_ready)
            self.thumbnail_loader.start()

    def on_preview_ready(self, file_path: str, pixmap: QPixmap):
        self.thumbnail_cache[file_path] = pixmap
        self.display_preview(file_path, pixmap)

    def display_preview(self, file_path: str, pixmap: QPixmap):
        self.preview_label.setPixmap(pixmap); self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_name = os.path.basename(file_path)
        try:
            size = os.path.getsize(file_path)
            size_text = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
            self.preview_info.setText(f"{file_name} • {pixmap.width()}x{pixmap.height()} • {size_text}")
        except Exception:
            self.preview_info.setText(file_name)

    # ---- Helpers for tree/results ----
    def clear_files(self):
        if self.thumbnail_loader: self.thumbnail_loader.stop()
        if self.folder_scanner: self.folder_scanner.stop()
        self.file_list.clear(); self.input_files = []; self.pre_existing_results = {}; self.failed_inputs = {}; self.output_files = {}; self.thumbnail_cache.clear()
        if hasattr(self, "unified_tree"): self.unified_tree.clear()
        self.preview_label.setText("Select an image to preview"); self.preview_label.setPixmap(QPixmap()); self.preview_info.setText("")
        self.input_count_label.setText("Input: 0"); self.output_count_label.setText("Output: 0")
        self.update_selection_info(); self.loading_label.setVisible(False); self.update_retry_button_state()

    def refresh_dataset_gallery(self):
        self.populate_input_tree()

    def restore_preexisting_results(self):
        if not self.pre_existing_results:
            return
        for input_path, output_path in self.pre_existing_results.items():
            if not output_path or not os.path.exists(output_path):
                continue
            if input_path in self.output_files and self.output_files[input_path] == output_path:
                continue
            self.add_result_to_tree(input_path, output_path)
        self.update_retry_button_state()

    def update_retry_button_state(self):
        if hasattr(self, "retry_failed_btn"):
            has_failed = bool(self.failed_inputs)
            self.retry_failed_btn.setEnabled(has_failed)
            self.retry_failed_btn.setToolTip("Retry processing for failed images." if has_failed else "No failed images to retry.")

    def remove_error_entry(self, image_path: str):
        if not hasattr(self, "output_root"):
            return
        error_dir = None
        for i in range(self.output_root.childCount()):
            child = self.output_root.child(i)
            if child.data(0, Qt.ItemDataRole.UserRole) == "errors" or child.text(0) == "❌ Errors":
                error_dir = child; break
        if not error_dir:
            return
        removed = False
        for j in range(error_dir.childCount()):
            item = error_dir.child(j)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, tuple) and data[0] == "error" and data[1] == image_path:
                error_dir.takeChild(j); removed = True; break
        if removed and error_dir.childCount() == 0:
            idx = self.output_root.indexOfChild(error_dir)
            if idx != -1:
                self.output_root.takeChild(idx)

    def set_error_entry_status(self, image_path: str, status: str, symbol: str):
        if not hasattr(self, "output_root"):
            return
        error_dir = None
        for i in range(self.output_root.childCount()):
            child = self.output_root.child(i)
            if child.data(0, Qt.ItemDataRole.UserRole) == "errors" or child.text(0) == "❌ Errors":
                error_dir = child; break
        if not error_dir:
            return
        for j in range(error_dir.childCount()):
            item = error_dir.child(j)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, tuple) and data[0] == "error" and data[1] == image_path:
                item.setText(1, status); item.setText(2, symbol); return

    def retry_failed_images(self):
        if not self.failed_inputs:
            QMessageBox.information(self, "No Failed Images", "There are no failed images to retry."); return
        existing, missing = [], []
        for path in list(self.failed_inputs.keys()):
            (existing if os.path.exists(path) else missing).append(path)
            if not os.path.exists(path):
                del self.failed_inputs[path]; self.remove_error_entry(path)
        if missing:
            missing_display = "\n".join(os.path.basename(p) or p for p in missing)
            QMessageBox.warning(self, "Missing Files", "The following failed images could not be retried because the files were not found:\n" + missing_display)
        if not existing:
            self.update_retry_button_state(); self.status_bar.showMessage("No failed images available for retry."); return
        for path in existing:
            self.update_input_status(path, "Retrying"); self.set_error_entry_status(path, "Retrying", "⏳")
        if hasattr(self, "retry_failed_btn"): self.retry_failed_btn.setEnabled(False)
        self.start_processing(image_paths=existing, preserve_outputs=True)

    def add_result_to_tree(self, input_path: str, output_path: str):
        self.output_files[input_path] = output_path
        if hasattr(self, "output_root"):
            output_dir = os.path.dirname(output_path); dir_item = None
            for i in range(self.output_root.childCount()):
                child = self.output_root.child(i)
                if child.data(0, Qt.ItemDataRole.UserRole) == output_dir:
                    dir_item = child; break
            if not dir_item:
                dir_item = QTreeWidgetItem(self.output_root); dir_item.setText(0, os.path.basename(output_dir) or "Output"); dir_item.setText(1, "Directory"); dir_item.setText(2, ""); dir_item.setData(0, Qt.ItemDataRole.UserRole, output_dir)
            file_item = QTreeWidgetItem(dir_item); file_item.setText(0, os.path.basename(output_path)); file_item.setText(1, "Processed"); file_item.setText(2, "✅"); file_item.setData(0, Qt.ItemDataRole.UserRole, ("output", output_path))
            self.output_root.setText(2, f"{len(self.output_files)} files"); self.output_root.setExpanded(True); dir_item.setExpanded(True)
            processed_count = len(self.output_files); self.output_count_label.setText(f"Output: {processed_count}")

    def get_expected_output_path(self, image_path: str, output_dir: Optional[str]) -> Optional[str]:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if output_dir:
            return os.path.join(output_dir, f"{image_name}_gemini.jpg")
        if output_dir == "":
            return os.path.join(os.getcwd(), f"{image_name}_gemini.jpg")
        default_dir = os.path.join(os.path.dirname(image_path), f"{image_name}_gemini_results")
        return os.path.join(default_dir, f"{image_name}_gemini.jpg")

    def filter_existing_outputs(self, image_paths, output_dir):
        pending, skipped = [], []
        for image_path in image_paths:
            expected_output = self.get_expected_output_path(image_path, output_dir)
            if expected_output and os.path.exists(expected_output):
                skipped.append((image_path, expected_output))
            else:
                pending.append(image_path)
        return pending, skipped

    def update_input_status(self, image_path: str, status: str):
        if not hasattr(self, "unified_tree"):
            return
        for i in range(self.unified_tree.topLevelItemCount()):
            root_item = self.unified_tree.topLevelItem(i)
            if root_item.data(0, Qt.ItemDataRole.UserRole) != "input_root":
                continue
            for dir_index in range(root_item.childCount()):
                dir_item = root_item.child(dir_index)
                for file_index in range(dir_item.childCount()):
                    file_item = dir_item.child(file_index)
                    data = file_item.data(0, Qt.ItemDataRole.UserRole)
                    if isinstance(data, tuple) and data[0] == "input" and data[1] == image_path:
                        file_item.setText(2, status); return

    def update_selection_info(self):
        count = self.file_list.count()
        if count == 0:
            self.selection_info.setText("No files selected")
        elif count == 1:
            input_filename = os.path.splitext(os.path.basename(self.file_list.item(0).text()))[0]
            self.selection_info.setText(f"1 file selected - output will be saved to '{input_filename}_gemini_results' folder")
        else:
            self.selection_info.setText(f"{count} files selected - batch processing with auto-generated output folder")

    def load_preset_prompt(self, preset_name):
        if not self.prompt_processor:
            QMessageBox.warning(self, "Error", "Prompt processor not available. Cannot load presets."); return
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
            prompt = self.prompt_processor.get_prompt(prompt_type); self.prompt_edit.setPlainText(prompt)
        except Exception as e:
            QMessageBox.warning(self, "Preset Error", f"Failed to load preset '{preset_name}' from JSON: {e}\n\nPlease check your prompts.json file.")

    # ---- Processing ----
    def start_processing(self, checked=False, *, image_paths=None, preserve_outputs=False):
        if not self.api_status:
            QMessageBox.warning(self, "API Error", "Gemini API is not connected. Please check your API key."); return
        if image_paths is None:
            image_paths = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        else:
            image_paths = list(dict.fromkeys(path for path in image_paths if path))
        if not image_paths:
            QMessageBox.warning(self, "No Files", "Please select image files to process."); return
        output_dir = None
        if self.auto_save.isChecked():
            if len(image_paths) == 1:
                input_dir = os.path.dirname(image_paths[0]); input_filename = os.path.splitext(os.path.basename(image_paths[0]))[0]
                output_dir = os.path.join(input_dir, f"{input_filename}_gemini_results")
            else:
                input_dirs = set(os.path.dirname(path) for path in image_paths)
                if len(input_dirs) == 1:
                    input_dir = list(input_dirs)[0]; output_dir = os.path.join(os.path.dirname(input_dir), f"{os.path.basename(input_dir)}_gemini_results")
                else:
                    first_input_dir = os.path.dirname(image_paths[0]); output_dir = os.path.join(os.path.dirname(first_input_dir), "gemini_results")
            os.makedirs(output_dir, exist_ok=True); print(f"📁 Auto-created output directory: {output_dir}"); self.status_bar.showMessage(f"Output directory created: {os.path.basename(output_dir)}")
        current_preset = self.preset_combo.currentText(); preset_to_type = {"Default Sky Removal": "default","Conservative Mode": "conservative","Aggressive Sky Removal": "aggressive","Building Preservation": "building_preservation","Urban Scene Focus": "urban_scene","Custom Template": "custom_template",}
        prompt_type = preset_to_type.get(current_preset, "default")
        current_prompt_text = self.prompt_edit.toPlainText().strip()
        default_prompt = ""
        if self.prompt_processor:
            try:
                default_prompt = self.prompt_processor.get_prompt(prompt_type)
            except Exception:
                default_prompt = ""
        prompt_kwargs = {}
        if current_prompt_text != default_prompt and current_prompt_text:
            prompt_type = "custom"; prompt_kwargs["custom_prompt"] = current_prompt_text
        if preserve_outputs:
            self.status_bar.showMessage("Retrying failed images...")
        else:
            self.clear_results(); self.restore_preexisting_results()
        pending_paths, skipped_outputs = self.filter_existing_outputs(image_paths, output_dir)
        skip_message = None
        if skipped_outputs:
            for input_path, output_path in skipped_outputs:
                self.update_input_status(input_path, "Skipped")
                if input_path in self.failed_inputs:
                    del self.failed_inputs[input_path]; self.remove_error_entry(input_path)
                if input_path not in self.output_files:
                    self.add_result_to_tree(input_path, output_path)
            self.update_retry_button_state()
            skipped_count = len(skipped_outputs); skip_message = f"Skipped {skipped_count} already processed image{'s' if skipped_count != 1 else ''}."; print(f"🔄 {skip_message}")
            if output_dir:
                self.status_bar.showMessage(f"{skip_message} • Output: {os.path.basename(output_dir)}")
            else:
                self.status_bar.showMessage(skip_message)
        if not pending_paths:
            if not skip_message:
                self.status_bar.showMessage("No images to process.")
            QMessageBox.information(self, "Nothing to Process", "All selected images already have Gemini results. Nothing new to process."); return
        image_paths = pending_paths
        for path in image_paths:
            self.update_input_status(path, "Queued")
        self.process_btn.setEnabled(False); self.cancel_btn.setVisible(True); self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        if self.tier_radio_t3.isChecked(): rate_limit_delay = self.tiers["tier3"]["delay"]
        elif self.tier_radio_t1.isChecked(): rate_limit_delay = self.tiers["tier1"]["delay"]
        else: rate_limit_delay = self.tiers["free"]["delay"]
        self.current_worker = ProcessingWorker(self.sky_remover, image_paths, output_dir or "", prompt_type, prompt_kwargs, float(rate_limit_delay))
        self.current_worker.progress_updated.connect(self.update_progress)
        self.current_worker.image_processed.connect(self.add_result)
        self.current_worker.error_occurred.connect(self.handle_error)
        self.current_worker.finished_all.connect(self.processing_finished)
        self.current_worker.start()

    def cancel_processing(self):
        if self.current_worker:
            self.current_worker.cancel(); self.status_bar.showMessage("Cancelling...")

    def update_progress(self, value, message):
        if value == -1:
            self.status_bar.showMessage(message)
        else:
            self.progress_bar.setValue(value); self.status_bar.showMessage(message)

    def add_result(self, input_path, output_path, pixmap):
        self.add_result_to_tree(input_path, output_path)
        if input_path in self.failed_inputs:
            del self.failed_inputs[input_path]; self.remove_error_entry(input_path); self.update_retry_button_state()
        self.update_input_status(input_path, "Done")
        if pixmap and not pixmap.isNull():
            self.thumbnail_cache[output_path] = pixmap

    def handle_error(self, error_msg, image_path):
        self.failed_inputs[image_path] = error_msg; self.update_input_status(image_path, "Failed"); self.remove_error_entry(image_path)
        if hasattr(self, "output_root"):
            error_dir = None
            for i in range(self.output_root.childCount()):
                child = self.output_root.child(i)
                if child.text(0) == "❌ Errors" or child.data(0, Qt.ItemDataRole.UserRole) == "errors":
                    error_dir = child; break
            if not error_dir:
                error_dir = QTreeWidgetItem(self.output_root); error_dir.setText(0, "❌ Errors"); error_dir.setText(1, "Folder"); error_dir.setText(2, ""); error_dir.setData(0, Qt.ItemDataRole.UserRole, "errors")
            file_item = QTreeWidgetItem(error_dir); file_item.setText(0, os.path.basename(image_path)); file_item.setText(1, "Failed"); file_item.setText(2, "❌"); file_item.setData(0, Qt.ItemDataRole.UserRole, ("error", image_path))
            for col in (0,1,2): file_item.setToolTip(col, error_msg)
            error_dir.setExpanded(True); self.output_root.setExpanded(True)
        self.update_retry_button_state()

    def processing_finished(self, results):
        self.process_btn.setEnabled(True); self.cancel_btn.setVisible(False); self.progress_bar.setVisible(False)
        success_count = sum(1 for r in results.values() if r.get("success", False)); total_count = len(results)
        self.status_bar.showMessage(f"Processing complete! {success_count}/{total_count} successful")
        QMessageBox.information(self, "Processing Complete", f"Successfully processed {success_count} out of {total_count} images.")
        self.current_worker = None; self.update_retry_button_state()

    def clear_results(self):
        if hasattr(self, "unified_tree"):
            if hasattr(self, "output_root"):
                while self.output_root.childCount() > 0:
                    self.output_root.takeChild(0)
                self.output_root.setText(2, "0 files")
        output_paths = list(self.output_files.values()); self.output_files = {}; self.failed_inputs = {}; self.output_count_label.setText("Output: 0"); self.update_retry_button_state()
        for path in output_paths:
            if path in self.thumbnail_cache:
                del self.thumbnail_cache[path]

    def show_about(self):
        QMessageBox.about(self, "About Gemini Sky Removal", "🦍 Gemini Sky Removal - Banana Nano Model\n\nA PyQt6 frontend for Google's Gemini 2.5 Flash API\nfor advanced sky and cloud removal from images.\n\nFeatures:\n• Single and batch image processing\n• Customizable prompts\n• Real-time preview\n• Professional UI\n\nBuilt with ❤️ using Gemini AI")

    def load_settings(self):
        geometry = self.settings.value("geometry")
        if geometry: self.restoreGeometry(geometry)
        saved_prompt = self.settings.value("prompt")
        if saved_prompt:
            old_hardcoded_start = "Please create a segmentation mask"
            if saved_prompt.startswith(old_hardcoded_start):
                self.settings.remove("prompt"); print("🧹 Cleared old hardcoded prompt from settings")
            else:
                self.prompt_edit.setPlainText(saved_prompt)
        saved_tier = self.settings.value("api_tier", "free")
        if saved_tier == "tier3": self.tier_radio_t3.setChecked(True)
        elif saved_tier == "tier1": self.tier_radio_t1.setChecked(True)
        else: self.tier_radio_free.setChecked(True)

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("prompt", self.prompt_edit.toPlainText())
        if self.tier_radio_t3.isChecked(): self.settings.setValue("api_tier", "tier3")
        elif self.tier_radio_t1.isChecked(): self.settings.setValue("api_tier", "tier1")
        else: self.settings.setValue("api_tier", "free")

    def closeEvent(self, event):
        self.save_settings()
        if self.current_worker: self.current_worker.cancel(); self.current_worker.wait()
        if self.thumbnail_loader: self.thumbnail_loader.stop(); self.thumbnail_loader.wait()
        if self.folder_scanner: self.folder_scanner.stop(); self.folder_scanner.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Gemini Sky Removal")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Banana Nano")
    window = GeminiSkyRemovalGUI(); window.show()
    sys.exit(app.exec())
