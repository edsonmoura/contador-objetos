from pathlib import Path
import traceback

import cv2
import numpy as np
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label

from object_counter.counter import count_by_label, format_counts
from object_counter.detector import YoloOnnxDetector, draw_detections


LOG_PATH = Path("app_error.log")


def write_error_log(error: BaseException) -> None:
    LOG_PATH.write_text("".join(traceback.format_exception(error)), encoding="utf-8")


class ObjectCounterLayout(BoxLayout):
    def __init__(self, **kwargs) -> None:
        super().__init__(orientation="vertical", spacing=8, padding=8, **kwargs)

        self.detector = self._create_detector()
        self.showing_result = False
        self.camera = None

        self.preview_area = FloatLayout()
        self.camera_error = self._create_camera()
        self.result_preview = Image(
            allow_stretch=True,
            keep_ratio=True,
            opacity=0,
            size_hint=(1, 1),
            pos_hint={"x": 0, "y": 0},
        )
        self.status = Label(
            text=self._initial_status(),
            size_hint_y=None,
            height=150,
            halign="left",
            valign="top",
        )
        self.status.bind(size=self._sync_label_text_size)

        self.button = Button(
            text="Contar",
            size_hint_y=None,
            height=56,
            on_press=self.count_objects,
        )

        if self.camera is None:
            self.preview_area.add_widget(
                Label(
                    text="Camera nao disponivel neste computador.",
                    halign="center",
                    valign="middle",
                )
            )
            self.button.disabled = True
        else:
            self.preview_area.add_widget(self.camera)

        self.preview_area.add_widget(self.result_preview)

        self.add_widget(self.preview_area)
        self.add_widget(self.button)
        self.add_widget(self.status)

    def _create_detector(self) -> YoloOnnxDetector | None:
        try:
            return YoloOnnxDetector()
        except (FileNotFoundError, cv2.error):
            return None

    def _create_camera(self) -> str | None:
        try:
            self.camera = Camera(
                play=True,
                resolution=(640, 480),
                size_hint=(1, 1),
                pos_hint={"x": 0, "y": 0},
            )
            return None
        except Exception as error:
            write_error_log(error)
            self.camera = None
            return str(error)

    def _initial_status(self) -> str:
        if self.camera_error:
            return "Camera nao disponivel no desktop. Teste com: python scripts/count_image.py sua_foto.jpg"

        model_path = Path("models/yolov8n.onnx")
        if model_path.exists():
            return "Aponte a camera para os objetos e toque em Contar."
        return "Modelo nao encontrado. Rode: python scripts/download_model.py"

    def _sync_label_text_size(self, instance: Label, size: tuple[int, int]) -> None:
        instance.text_size = size

    def count_objects(self, _button: Button) -> None:
        if self.showing_result:
            self.show_live_camera()
            return

        if self.detector is None:
            self.status.text = "Modelo nao carregado. Baixe o YOLOv8n ONNX primeiro."
            return

        frame = self._camera_texture_to_frame()
        if frame is None:
            self.status.text = "A camera ainda nao capturou uma imagem."
            return

        detections = self.detector.detect(frame)
        counts, total = count_by_label(detections)
        annotated_frame = draw_detections(frame, detections)

        self.result_preview.texture = self._frame_to_texture(annotated_frame)
        self.result_preview.opacity = 1
        self.showing_result = True
        self.button.text = "Voltar para camera"
        self.status.text = format_counts(counts, total)

    def _camera_texture_to_frame(self):
        if self.camera is None:
            return None

        texture = self.camera.texture
        if texture is None:
            return None

        width, height = map(int, texture.size)
        pixels = np.frombuffer(texture.pixels, dtype=np.uint8)
        rgba = pixels.reshape(height, width, 4)
        rgba = np.flipud(rgba)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def _frame_to_texture(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        flipped = cv2.flip(frame_rgb, 0)
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt="rgb")
        texture.blit_buffer(flipped.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        return texture

    def show_live_camera(self) -> None:
        if self.showing_result:
            self.result_preview.opacity = 0
            self.showing_result = False
            self.button.text = "Contar"
            self.status.text = "Aponte a camera para os objetos e toque em Contar."


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
