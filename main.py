from pathlib import Path
import traceback

import cv2
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label

from object_counter.blob_counter import (
    detect_contrast_objects,
    detect_dark_objects,
    detect_light_objects,
)
from object_counter.counter import count_by_label, format_counts
from object_counter.detector import YoloOnnxDetector, draw_detections


LOG_PATH = Path("app_error.log")

MODES = {
    "beans": ("Feijao", "feijao", detect_dark_objects, 250),
    "rice": ("Arroz", "arroz", detect_light_objects, 40),
    "parts": ("Pecas", "peca", detect_contrast_objects, 120),
    "dark": ("Escuros", "objeto escuro", detect_dark_objects, 800),
    "yolo": ("YOLO", None, None, None),
}


def write_error_log(error: BaseException) -> None:
    LOG_PATH.write_text("".join(traceback.format_exception(error)), encoding="utf-8")


def request_android_camera_permission() -> None:
    try:
        from android.permissions import Permission, request_permissions

        request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])
    except Exception:
        pass


class ObjectCounterLayout(BoxLayout):
    def __init__(self, **kwargs) -> None:
        super().__init__(orientation="vertical", spacing=8, padding=8, **kwargs)

        request_android_camera_permission()

        self.mode = "beans"
        self.detector = self._create_detector()
        self.last_photo_path = None

        self.preview = Image(allow_stretch=True, keep_ratio=True)
        self.status = Label(
            text="Escolha o modo e toque em Tirar foto.",
            size_hint_y=None,
            height=150,
            halign="left",
            valign="top",
        )
        self.status.bind(size=self._sync_label_text_size)

        self.mode_buttons = GridLayout(cols=5, size_hint_y=None, height=52, spacing=4)
        for mode_key, mode_info in MODES.items():
            button = Button(text=mode_info[0])
            button.bind(on_press=lambda _button, key=mode_key: self.set_mode(key))
            self.mode_buttons.add_widget(button)

        self.capture_button = Button(
            text="Tirar foto e contar",
            size_hint_y=None,
            height=56,
            on_press=self.take_photo,
        )

        self.add_widget(self.preview)
        self.add_widget(self.mode_buttons)
        self.add_widget(self.capture_button)
        self.add_widget(self.status)

    def _create_detector(self) -> YoloOnnxDetector | None:
        try:
            return YoloOnnxDetector()
        except (FileNotFoundError, cv2.error):
            return None

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.status.text = f"Modo selecionado: {MODES[mode][0]}"

    def take_photo(self, _button: Button) -> None:
        output_path = Path(App.get_running_app().user_data_dir) / "capture.jpg"
        self.last_photo_path = output_path

        try:
            from plyer import camera

            camera.take_picture(str(output_path), self.on_photo_taken)
            self.status.text = "Camera aberta. Tire a foto e confirme."
        except Exception as error:
            write_error_log(error)
            self.status.text = (
                "Nao foi possivel abrir a camera nativa.\n"
                f"{type(error).__name__}: {error}\n"
                f"Detalhes em: {LOG_PATH}"
            )

    def on_photo_taken(self, photo_path: str | None) -> None:
        path = Path(photo_path or self.last_photo_path or "")
        if not path.exists():
            self.status.text = "Foto nao encontrada. Tente novamente."
            return

        self.analyze_image(path)

    def analyze_image(self, image_path: Path) -> None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            self.status.text = f"Nao foi possivel abrir a imagem: {image_path}"
            return

        try:
            detections = self.detect(frame)
            counts, total = count_by_label(detections)
            annotated = draw_detections(frame, detections)
            self.preview.texture = self._frame_to_texture(annotated)
            self.status.text = format_counts(counts, total)
        except Exception as error:
            write_error_log(error)
            self.status.text = f"Erro ao contar: {type(error).__name__}: {error}"

    def detect(self, frame):
        if self.mode == "yolo":
            if self.detector is None:
                raise RuntimeError("Modelo YOLO ONNX nao encontrado no APK.")
            return self.detector.detect(frame)

        _title, label, detector_function, min_area = MODES[self.mode]
        return detector_function(frame, min_area=min_area, label=label)

    def _frame_to_texture(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        flipped = cv2.flip(frame_rgb, 0)
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt="rgb")
        texture.blit_buffer(flipped.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        return texture

    def _sync_label_text_size(self, instance: Label, size: tuple[int, int]) -> None:
        instance.text_size = size


class ObjectCounterApp(App):
    title = "Contador de Objetos"

    def build(self):
        try:
            return ObjectCounterLayout()
        except Exception as error:
            write_error_log(error)
            return ErrorLayout(error)


class ErrorLayout(BoxLayout):
    def __init__(self, error: BaseException, **kwargs) -> None:
        super().__init__(orientation="vertical", padding=12, spacing=8, **kwargs)
        label = Label(
            text=(
                "O app encontrou um erro ao iniciar.\n\n"
                f"{type(error).__name__}: {error}\n\n"
                f"Detalhes salvos em: {LOG_PATH}"
            ),
            halign="left",
            valign="top",
        )
        label.bind(size=self._sync_label_text_size)
        self.add_widget(label)

    def _sync_label_text_size(self, instance: Label, size: tuple[int, int]) -> None:
        instance.text_size = size


if __name__ == "__main__":
    try:
        ObjectCounterApp().run()
    except Exception as error:
        write_error_log(error)
        raise
