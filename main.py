import os
import time
import numpy as np
import cv2
import onnxruntime as ort

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from insightface.app import FaceAnalysis


# -------------------- CONFIG --------------------
CAM_ID = 0

DB_PATH = "faces_db.npz"
LIVENESS_ONNX = "best_model.onnx"
REAL_INDEX = 0

LIVE_THRESHOLD = 0.85
COS_THRESHOLD = 0.35

ENROLL_SAMPLES = 15
ENROLL_MIN_SAMPLES = 6
NEEDED_CONSECUTIVE_ACCEPTS = 3
# ------------------------------------------------


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


class FaceDB:
    def __init__(self, path: str):
        self.path = path
        self.names = []
        self.embs = np.zeros((0, 512), dtype=np.float32)
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        data = np.load(self.path, allow_pickle=True)
        self.names = data["names"].tolist()
        self.embs = data["embs"].astype(np.float32)

    def save(self):
        np.savez(self.path, names=np.array(self.names, dtype=object), embs=self.embs.astype(np.float32))

    def add(self, name: str, emb: np.ndarray):
        self.names.append(name)
        self.embs = np.vstack([self.embs, emb.astype(np.float32)[None, :]]).astype(np.float32)
        self.save()

    def clear(self):
        self.names = []
        self.embs = np.zeros((0, 512), dtype=np.float32)
        self.save()

    def identify(self, emb: np.ndarray):
        if self.embs.shape[0] == 0:
            return None, 0.0
        sims = np.array([cosine_sim(emb, e) for e in self.embs], dtype=np.float32)
        idx = int(np.argmax(sims))
        return self.names[idx], float(sims[idx])


class AntiSpoofONNX:

    def __init__(self, onnx_path: str, real_index: int = 1, debug: bool = False):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX liveness model not found: {onnx_path}")

        self.real_index = int(real_index)
        self.debug = debug

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

        in_shape = self.sess.get_inputs()[0].shape  # [1,3,H,W] обычно
        self.in_h = int(in_shape[2]) if isinstance(in_shape[2], (int, np.integer)) else 128
        self.in_w = int(in_shape[3]) if isinstance(in_shape[3], (int, np.integer)) else 128

    def predict_live(self, face_bgr: np.ndarray) -> tuple[float, int]:
        """
        Returns:
          prob_real (float), pred_class (int)
        """
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)

        x = rgb.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,H,W)

        out = self.sess.run(None, {self.input_name: x})
        y = np.array(out[0]).reshape(-1)

        # Если модель вернула уже prob_real
        if y.size == 1:
            prob_real = float(np.clip(y[0], 0.0, 1.0))
            pred = 1 if prob_real >= 0.5 else 0
            return prob_real, pred

        # Логиты -> softmax
        p = softmax(y)
        pred = int(np.argmax(p))

        # вероятность именно класса REAL
        if 0 <= self.real_index < p.size:
            prob_real = float(p[self.real_index])
        else:
            # если real_index некорректный
            prob_real = float(np.clip(np.max(p), 0.0, 1.0))

        if self.debug:
            print("liveness logits:", y, "probs:", p, "pred:", pred, "prob_real:", prob_real)

        return float(np.clip(prob_real, 0.0, 1.0)), pred


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Access (ReID + Anti-spoof)")

        # Core
        self.db = FaceDB(DB_PATH)
        self.antispoof = AntiSpoofONNX(LIVENESS_ONNX, real_index=REAL_INDEX, debug=False)

        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        self.cap = cv2.VideoCapture(CAM_ID)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.mode = "verify"
        self.accept_streak = 0

        self.enroll_name = ""
        self.enroll_collected = []

        self.video_label = tk.Label(root)
        self.video_label.grid(row=0, column=0, columnspan=4, padx=8, pady=8)

        tk.Label(root, text="Name:").grid(row=1, column=0, sticky="e", padx=6)
        self.name_entry = tk.Entry(root, width=20)
        self.name_entry.grid(row=1, column=1, sticky="w", padx=6)

        self.enroll_btn = tk.Button(root, text="Enroll", command=self.start_enroll)
        self.enroll_btn.grid(row=1, column=2, padx=6)

        self.verify_btn = tk.Button(root, text="Verify (toggle)", command=self.toggle_verify)
        self.verify_btn.grid(row=1, column=3, padx=6)

        self.clear_btn = tk.Button(root, text="Clear DB", command=self.clear_db)
        self.clear_btn.grid(row=2, column=0, padx=6, pady=6, sticky="w")

        self.status = tk.StringVar()
        self.status.set(self._status_text())
        self.status_label = tk.Label(root, textvariable=self.status, anchor="w", justify="left")
        self.status_label.grid(row=2, column=1, columnspan=3, sticky="w", padx=6)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._loop()

    def _status_text(self):
        return (
            f"DB size: {len(self.db.names)} | mode: {self.mode} | "
            f"live_th={LIVE_THRESHOLD:.2f} cos_th={COS_THRESHOLD:.2f} real_idx={REAL_INDEX}"
        )

    def clear_db(self):
        if messagebox.askyesno("Clear DB", "Удалить все лица из базы?"):
            self.db.clear()
            self.status.set(self._status_text())

    def start_enroll(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Enroll", "Введи имя (например: ivan)")
            return
        self.mode = "enroll"
        self.enroll_name = name
        self.enroll_collected = []
        self.accept_streak = 0
        self.status.set(f"ENROLL: {name} | collected 0/{ENROLL_SAMPLES} (need live>= {LIVE_THRESHOLD:.2f})")

    def toggle_verify(self):
        self.mode = "verify"
        self.enroll_name = ""
        self.enroll_collected = []
        self.accept_streak = 0
        self.status.set(self._status_text())

    def _pick_main_face(self, faces):
        if not faces:
            return None

        def area(f):
            x1, y1, x2, y2 = f.bbox.astype(int)
            return max(0, x2 - x1) * max(0, y2 - y1)

        return max(faces, key=area)

    def _loop(self):
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(10, self._loop)
            return

        faces = self.face_app.get(frame)
        f = self._pick_main_face(faces)

        vis = frame.copy()
        decision_line = ""
        color = (0, 0, 255)

        if f is not None:
            x1, y1, x2, y2 = f.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(vis.shape[1] - 1, x2), min(vis.shape[0] - 1, y2)

            face_crop = frame[y1:y2, x1:x2]

            live_prob = 0.0
            live_pred = -1
            if face_crop.size > 0:
                live_prob, live_pred = self.antispoof.predict_live(face_crop)

            emb = f.embedding.astype(np.float32)
            name, sim = self.db.identify(emb)

            live_ok = (live_pred == self.antispoof.real_index) and (live_prob >= LIVE_THRESHOLD)

            if self.mode == "verify":
                id_ok = (name is not None) and (sim >= COS_THRESHOLD)

                if live_ok and id_ok:
                    self.accept_streak += 1
                else:
                    self.accept_streak = 0

                if self.accept_streak >= NEEDED_CONSECUTIVE_ACCEPTS:
                    decision_line = f"✅ ACCESS GRANTED: {name} | sim={sim:.2f} live={live_prob:.2f} pred={live_pred}"
                    color = (0, 255, 0)
                else:
                    decision_line = f"❌ DENIED | best={name} sim={sim:.2f} live={live_prob:.2f} pred={live_pred}"
                    color = (0, 0, 255)

            elif self.mode == "enroll":
                if live_ok:
                    self.enroll_collected.append(emb)
                    decision_line = (
                        f"ENROLL {self.enroll_name}: {len(self.enroll_collected)}/{ENROLL_SAMPLES} "
                        f"live={live_prob:.2f} pred={live_pred}"
                    )
                    color = (0, 255, 0)
                    time.sleep(0.06)
                else:
                    decision_line = (
                        f"ENROLL {self.enroll_name}: waiting for LIVE face... "
                        f"live={live_prob:.2f} pred={live_pred}"
                    )
                    color = (0, 165, 255)

                self.status.set(decision_line)

                if len(self.enroll_collected) >= ENROLL_SAMPLES:
                    if len(self.enroll_collected) < ENROLL_MIN_SAMPLES:
                        messagebox.showerror("Enroll", "Слишком мало качественных кадров. Попробуй ближе/больше света.")
                    else:
                        mean_emb = l2_normalize(np.mean(np.stack(self.enroll_collected), axis=0)).astype(np.float32)
                        self.db.add(self.enroll_name, mean_emb)
                        messagebox.showinfo("Enroll", f"Saved: {self.enroll_name}\nDB size={len(self.db.names)}")
                    self.toggle_verify()

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            if decision_line and self.mode == "verify":
                self.status.set(self._status_text() + " | " + decision_line)

        if self.mode == "verify":
            cv2.putText(vis, self.status.get()[:160], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        else:
            cv2.putText(vis, decision_line[:160], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self._loop)

    def on_close(self):
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    if not os.path.exists(LIVENESS_ONNX):
        raise SystemExit(
            f"Не найден {LIVENESS_ONNX}.\n"
            "Положи ONNX анти-спуфинг рядом с этим файлом и укажи путь в LIVENESS_ONNX."
        )

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
