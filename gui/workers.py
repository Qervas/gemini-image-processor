import os
from typing import List, Optional
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage

class FolderScannerThread(QThread):
    files_found = pyqtSignal(list)  # List[str]
    scan_finished = pyqtSignal(int)

    def __init__(self, folder_path: str):
        super().__init__()
        self.folder_path = folder_path
        self.is_cancelled = False
        self.already_processed = []  # list[(input, output)]

    def stop(self):
        self.is_cancelled = True

    @staticmethod
    def _is_result_directory(directory_path: str) -> bool:
        name = os.path.basename(directory_path.rstrip(os.sep)).lower()
        return name.endswith("_gemini_results") or "gemini_results" in name

    @staticmethod
    def _is_result_file(filename: str) -> bool:
        return "gemini" in filename.lower()

    @staticmethod
    def _find_existing_result(image_path: str) -> Optional[str]:
        image_dir = os.path.dirname(image_path)
        image_name, _ = os.path.splitext(os.path.basename(image_path))
        candidate_dirs = {
            image_dir,
            os.path.join(image_dir, f"{image_name}_gemini_results"),
        }
        parent_dir = os.path.dirname(image_dir)
        if parent_dir:
            dir_name = os.path.basename(image_dir)
            candidate_dirs.add(os.path.join(parent_dir, f"{dir_name}_gemini_results"))
            candidate_dirs.add(os.path.join(parent_dir, "gemini_results"))
        for directory in list(candidate_dirs):
            if not directory:
                continue
            stem = os.path.join(directory, f"{image_name}_gemini")
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidate = stem + ext
                if os.path.exists(candidate):
                    return candidate
        return None

    def run(self):
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        image_files = []
        skipped_dir_count = 0
        skipped_file_count = 0
        already_processed_count = 0
        try:
            for root, dirs, files in os.walk(self.folder_path):
                if self.is_cancelled:
                    break
                if self._is_result_directory(root):
                    skipped_dir_count += 1
                    dirs[:] = []
                    continue
                original_dirs = list(dirs)
                dirs[:] = [d for d in dirs if not self._is_result_directory(os.path.join(root, d))]
                skipped_dir_count += len(original_dirs) - len(dirs)
                for file in files:
                    if self.is_cancelled:
                        break
                    if self._is_result_file(file):
                        skipped_file_count += 1
                        continue
                    if not any(file.lower().endswith(ext) for ext in image_extensions):
                        continue
                    full_path = os.path.join(root, file)
                    existing_result = self._find_existing_result(full_path)
                    if existing_result:
                        already_processed_count += 1
                        self.already_processed.append((full_path, existing_result))
                        continue
                    image_files.append(full_path)
            if not self.is_cancelled:
                self.files_found.emit(image_files)
                self.scan_finished.emit(len(image_files) + already_processed_count)
        except Exception as e:
            print(f"Error scanning folder: {e}")
            self.scan_finished.emit(0)

class SingleThumbnailLoader(QThread):
    thumbnail_ready = pyqtSignal(str, QPixmap)

    def __init__(self, image_path: str, thumb_size: int = 300):
        super().__init__()
        self.image_path = image_path
        self.thumb_size = thumb_size
        self.is_cancelled = False

    def stop(self):
        self.is_cancelled = True

    def run(self):
        if self.is_cancelled:
            return
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.thumb_size,
                    self.thumb_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.thumbnail_ready.emit(self.image_path, scaled)
            else:
                error_pixmap = QPixmap(self.thumb_size, self.thumb_size)
                error_pixmap.fill(Qt.GlobalColor.lightGray)
                self.thumbnail_ready.emit(self.image_path, error_pixmap)
        except Exception as e:
            print(f"Error loading thumbnail for {self.image_path}: {e}")
            error_pixmap = QPixmap(self.thumb_size, self.thumb_size)
            error_pixmap.fill(Qt.GlobalColor.red)
            self.thumbnail_ready.emit(self.image_path, error_pixmap)

class ProcessingWorker(QThread):
    progress_updated = pyqtSignal(int, str)
    image_processed = pyqtSignal(str, str, QPixmap)
    finished_all = pyqtSignal(dict)
    error_occurred = pyqtSignal(str, str)

    def __init__(self, sky_remover, image_paths: List[str], output_dir: str,
                 prompt_type: str, prompt_kwargs: dict | None = None,
                 rate_limit_delay: float = 6.0):
        super().__init__()
        self.sky_remover = sky_remover
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.prompt_type = prompt_type
        self.prompt_kwargs = prompt_kwargs or {}
        self.is_cancelled = False
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        self.retry_count = 0
        self.max_retries = 3

    def cancel(self):
        self.is_cancelled = True

    def _sleep_with_cancel(self, total_seconds, message=None, poll_interval=0.2):
        import time
        if total_seconds <= 0:
            return True
        if message:
            self.progress_updated.emit(-1, message)
        end_time = time.time() + total_seconds
        while not self.is_cancelled:
            remaining = end_time - time.time()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval, remaining))
        return not self.is_cancelled

    def _wait_for_rate_limit(self):
        import time
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last_request
            message = f"⏱️  Rate limiting: waiting {wait_time:.1f}s before next request..."
            if not self._sleep_with_cancel(wait_time, message):
                return False
        self.last_request_time = time.time()
        return True

    def _handle_rate_limit_error(self, error, image_path):
        error_str = str(error).lower()
        if ("rate limit" in error_str) or ("quota exceeded" in error_str) or ("429" in error_str):
            self.retry_count += 1
            if self.retry_count <= self.max_retries:
                backoff_time = 30 * (2 ** (self.retry_count - 1))
                message = f"🚦 Rate limit hit! Retrying in {backoff_time}s (attempt {self.retry_count}/{self.max_retries})..."
                if not self._sleep_with_cancel(backoff_time, message, poll_interval=0.5):
                    return False
                return True
            else:
                self.error_occurred.emit(
                    f"Rate limit exceeded after {self.max_retries} retries. Please wait before trying again.",
                    image_path,
                )
                return False
        return False

    def run(self):
        results = {}
        total = len(self.image_paths)
        for i, image_path in enumerate(self.image_paths, 1):
            if self.is_cancelled:
                break
            self.retry_count = 0
            while self.retry_count <= self.max_retries:
                try:
                    if i > 1:
                        if not self._wait_for_rate_limit():
                            break
                        if self.is_cancelled:
                            break
                    self.progress_updated.emit(int((i - 1) / total * 100), f"Processing {i}/{total}: {os.path.basename(image_path)}")
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = os.path.join(self.output_dir, f"{image_name}_gemini.jpg")
                    result = self.sky_remover.gemini_sky_removal(
                        image_path,
                        output_path,
                        prompt_type=self.prompt_type,
                        **self.prompt_kwargs,
                    )
                    height, width, channel = result.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(result.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.image_processed.emit(image_path, output_path, pixmap)
                    results[image_path] = {"success": True, "output_path": output_path}
                    break
                except Exception as e:
                    if self.is_cancelled:
                        break
                    handled = self._handle_rate_limit_error(e, image_path)
                    if self.is_cancelled:
                        break
                    if handled:
                        continue
                    error_msg = f"Failed to process {os.path.basename(image_path)}: {str(e)}"
                    self.error_occurred.emit(error_msg, image_path)
                    results[image_path] = {"success": False, "error": str(e)}
                    break
            if self.is_cancelled:
                break
        self.progress_updated.emit(100, "Processing complete!")
        self.finished_all.emit(results)
