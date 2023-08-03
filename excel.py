import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtCore import pyqtSignal, QObject, QRunnable, QThreadPool

class LabelingProcessor(QObject):
    review_updated = pyqtSignal(str)
    progress_saved = pyqtSignal()

    def __init__(self, excel_file_path, output_file_path):
        super().__init__()
        self.df = pd.read_excel(excel_file_path)
        if 'label' not in self.df.columns:
            self.df['label'] = None

        self.total_reviews = len(self.df)
        self.current_index = 0
        self.output_file = output_file_path

    def save_progress(self, label):
        self.df.at[self.current_index, 'label'] = int(label)
        self.df.to_excel(self.output_file, index=False, engine='openpyxl')
        self.current_index += 1

    def get_next_review(self):
        if self.current_index >= self.total_reviews:
            self.progress_saved.emit()
            return "All reviews have been labeled."

        review = self.df.at[self.current_index, 'review_description']
        return f"Review: {review}"

class LabelingWorker(QObject, QRunnable):
    finished = pyqtSignal()  # Custom signal to indicate worker completion

    def __init__(self, processor, label):
        super().__init__()
        self.processor = processor
        self.label = label

    def run(self):
        self.processor.save_progress(self.label)
        self.finished.emit()  # Emit the custom signal to indicate worker completion

class AppReviewLabelingUI(QMainWindow):
    next_review_requested = pyqtSignal()

    def __init__(self, excel_file_path, output_file_path):
        super().__init__()
        self.processor = LabelingProcessor(excel_file_path, output_file_path)
        self.thread_pool = QThreadPool()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("App Review Labeling Tool")

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.review_label = QLabel("Review: ")
        layout.addWidget(self.review_label)

        self.label_var = QLineEdit()
        layout.addWidget(self.label_var)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.on_next_button_clicked)
        layout.addWidget(self.next_button)

        self.message_label = QLabel()
        layout.addWidget(self.message_label)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.quit_labeling)
        layout.addWidget(self.quit_button)

        self.next_review_requested.connect(self.show_next_review)
        self.next_review_requested.emit()

    def show_next_review(self):
        review = self.processor.get_next_review()
        if review == "All reviews have been labeled.":
            self.next_button.setEnabled(False)
            self.quit_button.setText("Quit and Save Progress")
        self.review_label.setText(review)
        self.label_var.setText('')
        self.message_label.setText('')
        self.next_button.setEnabled(True)

    def on_next_button_clicked(self):
        label = self.label_var.text()
        if label.isdigit():
            self.message_label.setText('')
            self.next_button.setEnabled(False)

            worker = LabelingWorker(self.processor, label)
            worker.finished.connect(self.on_worker_finished, type=QObject.Qt.QueuedConnection)  # Connect the custom signal to the slot
            self.thread_pool.start(worker)

            self.next_review_requested.emit()
        else:
            self.message_label.setText("Please enter a valid score (0 or 1).")

    def on_worker_finished(self):
        self.next_button.setEnabled(True)  # Enable the Next button after worker finishes

    def quit_labeling(self):
        if self.message_label.text() == "All reviews have been labeled.":
            self.close()
        else:
            if self.confirm_quit():
                self.message_label.setText("Saving progress...")
                self.processor.save_progress(self.label_var.text())
                self.message_label.setText("Your progress has been saved.")
                self.next_button.setEnabled(False)
                self.quit_button.setEnabled(False)

    def confirm_quit(self):
        result = QMessageBox.question(self, "Quit Labeling", "Do you want to save progress and quit?",
                                      QMessageBox.Yes | QMessageBox.No)
        return result == QMessageBox.Yes

if __name__ == "__main__":
    excel_file_path = "nagad.xlsx"
    output_file_path = "labeled_nagad_progress.xlsx"

    app = QApplication(sys.argv)
    ui = AppReviewLabelingUI(excel_file_path, output_file_path)
    ui.show()
    sys.exit(app.exec_())
