import os
import time
import argparse
import numpy as np
import cv2
import onnxruntime as ort


from insightface.app import FaceAnalysis


# -------------------- CONFIG --------------------
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
            # Scalar output means the model already returned probability of REAL.
            # Keep pred consistent with configured real_index; otherwise with REAL_INDEX=0
            # live_ok can never become True when prob_real is high.
            prob_real = float(np.clip(y[0], 0.0, 1.0))
            pred = self.real_index if prob_real >= 0.5 else 1 - self.real_index
            if self.debug:
                print("liveness scalar:", y, "pred:", pred, "prob_real:", prob_real)
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


class VideoFileProcessor:
    def __init__(
        self,
        video_path: str,
        mode: str = "verify",
        enroll_name: str = "",
        display: bool = True,
        debug_liveness: bool = False,
        strict_liveness: bool = False,
    ):
        # Core
        self.db = FaceDB(DB_PATH)
        self.antispoof = AntiSpoofONNX(LIVENESS_ONNX, real_index=REAL_INDEX, debug=debug_liveness)

        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        self.video_path = video_path
        self.mode = mode
        self.display = display
        self.accept_streak = 0
        self.access_granted = False

        self.enroll_name = enroll_name
        self.enroll_collected = []
        self.live_seen = 0
        self.max_live_prob = 0.0
        self.debug_liveness = debug_liveness
        # Для видеофайлов антиспуфинг часто распознаёт запись как replay-attack.
        # Поэтому по умолчанию НЕ блокируем enroll/verify по liveness, но score всё равно считаем
        # и показываем. Старое строгое поведение можно вернуть флагом --strict-liveness.
        self.strict_liveness = bool(strict_liveness)

        if self.mode == "enroll" and not self.enroll_name:
            raise ValueError("Для enroll нужно передать имя: --enroll <name>")

    def _status_text(self):
        return (
            f"DB size: {len(self.db.names)} | mode: {self.mode} | "
            f"live_th={LIVE_THRESHOLD:.2f} cos_th={COS_THRESHOLD:.2f} real_idx={REAL_INDEX} | "
            f"liveness_gate={'ON' if self.strict_liveness else 'OFF(video bypass)'}"
        )

    def _pick_main_face(self, faces):
        if not faces:
            return None

        def area(f):
            x1, y1, x2, y2 = f.bbox.astype(int)
            return max(0, x2 - x1) * max(0, y2 - y1)

        return max(faces, key=area)

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str, tuple[int, int, int]]:
        faces = self.face_app.get(frame)
        f = self._pick_main_face(faces)

        vis = frame.copy()
        decision_line = "No face detected"
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

            self.max_live_prob = max(self.max_live_prob, live_prob)
            raw_live_ok = (live_pred == self.antispoof.real_index) and (live_prob >= LIVE_THRESHOLD)
            # Если strict_liveness выключен, liveness используется только как диагностический score,
            # а не как стоп-фактор. Это нужно, чтобы обычные видеофайлы проходили enroll/verify.
            live_ok = raw_live_ok if self.strict_liveness else True
            if raw_live_ok:
                self.live_seen += 1

            if self.mode == "verify":
                id_ok = (name is not None) and (sim >= COS_THRESHOLD)

                if live_ok and id_ok:
                    self.accept_streak += 1
                else:
                    self.accept_streak = 0

                if self.accept_streak >= NEEDED_CONSECUTIVE_ACCEPTS:
                    self.access_granted = True
                    decision_line = f"ACCESS GRANTED: {name} | sim={sim:.2f} live={live_prob:.2f} pred={live_pred}{'' if self.strict_liveness else ' BYPASS'}"
                    color = (0, 255, 0)
                else:
                    decision_line = f"DENIED | best={name} sim={sim:.2f} live={live_prob:.2f} pred={live_pred}{'' if self.strict_liveness else ' BYPASS'}"
                    color = (0, 0, 255)

            elif self.mode == "enroll":
                if live_ok:
                    self.enroll_collected.append(emb)
                    decision_line = (
                        f"ENROLL {self.enroll_name}: {len(self.enroll_collected)}/{ENROLL_SAMPLES} "
                        f"live={live_prob:.2f} pred={live_pred}{'' if self.strict_liveness else ' BYPASS'}"
                    )
                    color = (0, 255, 0)
                    time.sleep(0.06)
                else:
                    decision_line = (
                        f"ENROLL {self.enroll_name}: waiting for LIVE face... "
                        f"live={live_prob:.2f} pred={live_pred}{'' if self.strict_liveness else ' BYPASS'}"
                    )
                    color = (0, 165, 255)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        cv2.putText(vis, decision_line[:160], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return vis, decision_line, color

    def run(self):
        frame_idx = 0
        last_line = ""
        saved = False

        print(self._status_text())
        if self.mode == "enroll":
            if self.strict_liveness:
                print(f"ENROLL: {self.enroll_name} | collected 0/{ENROLL_SAMPLES} (need live>= {LIVE_THRESHOLD:.2f})")
            else:
                print(f"ENROLL: {self.enroll_name} | collected 0/{ENROLL_SAMPLES} (video liveness bypass ON)")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame_idx += 1
            vis, decision_line, _ = self._process_frame(frame)
            last_line = decision_line

            if frame_idx == 1 or frame_idx % 10 == 0 or self.mode == "enroll":
                print(f"frame={frame_idx}: {decision_line}")

            if self.display:
                cv2.imshow("Face Access (video file)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if self.mode == "enroll" and len(self.enroll_collected) >= ENROLL_SAMPLES:
                saved = self._save_enroll()
                break

        if self.mode == "enroll" and not saved:
            print(
                "Enroll failed: слишком мало качественных кадров. "
                f"Collected {len(self.enroll_collected)}, required {ENROLL_SAMPLES}. "
                f"live_seen={self.live_seen}, max_live={self.max_live_prob:.3f}, "
                f"threshold={LIVE_THRESHOLD:.3f}, real_idx={REAL_INDEX}."
            )

        self.cap.release()
        if self.display:
            cv2.destroyAllWindows()

        print(f"DONE | frames={frame_idx} | last='{last_line}'")
        return saved if self.mode == "enroll" else self.access_granted

    def _save_enroll(self) -> bool:
        if len(self.enroll_collected) < ENROLL_SAMPLES:
            print(
                "Enroll failed: слишком мало качественных кадров. "
                f"Collected {len(self.enroll_collected)}, required {ENROLL_SAMPLES}. "
                f"live_seen={self.live_seen}, max_live={self.max_live_prob:.3f}, "
                f"threshold={LIVE_THRESHOLD:.3f}, real_idx={REAL_INDEX}."
            )
            return False

        mean_emb = l2_normalize(np.mean(np.stack(self.enroll_collected), axis=0)).astype(np.float32)
        self.db.add(self.enroll_name, mean_emb)
        print(f"Enroll saved: {self.enroll_name} | DB size={len(self.db.names)}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Face Access: verify/enroll from a video file")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--enroll", metavar="NAME", help="Enroll NAME from the input video")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV preview window")
    parser.add_argument("--debug-liveness", action="store_true", help="Print raw liveness model outputs")
    parser.add_argument(
        "--strict-liveness",
        action="store_true",
        help="Require anti-spoofing to pass. By default video files bypass liveness gating.",
    )
    args = parser.parse_args()

    if not os.path.exists(LIVENESS_ONNX):
        raise SystemExit(
            f"Не найден {LIVENESS_ONNX}.\n"
            "Положи ONNX анти-спуфинг рядом с этим файлом и укажи путь в LIVENESS_ONNX."
        )

    if not os.path.exists(args.video):
        raise SystemExit(f"Не найден видеофайл: {args.video}")

    mode = "enroll" if args.enroll else "verify"
    processor = VideoFileProcessor(
        video_path=args.video,
        mode=mode,
        enroll_name=args.enroll or "",
        display=not args.no_display,
        debug_liveness=args.debug_liveness,
        strict_liveness=args.strict_liveness,
    )
    ok = processor.run()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
