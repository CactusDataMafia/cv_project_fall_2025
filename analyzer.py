import cv2
import numpy as np
import math
import pandas as pd
import mediapipe as mp


def parse_video(videopath):
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не получается открыть файл {videopath}")
    return cap


def count_l2_metrics(dict_of_metrics: dict) -> tuple:
    pas = (dict_of_metrics["mouthSmileLeft"] + dict_of_metrics["mouthSmileRight"] 
    + dict_of_metrics["mouthDimpleLeft"] + dict_of_metrics["mouthDimpleRight"])

    nas = (dict_of_metrics["mouthFrownLeft"] + dict_of_metrics["mouthFrownRight"]
    + dict_of_metrics["mouthPressLeft"] + dict_of_metrics["mouthPressRight"]
    + dict_of_metrics["noseSneerLeft"] + dict_of_metrics["noseSneerRight"])

    si = dict_of_metrics["browInnerUp"] + dict_of_metrics["browOuterUpLeft"] + dict_of_metrics["browOuterUpRight"] +dict_of_metrics["eyeWideLeft"] + dict_of_metrics["eyeWideRight"]

    epi = (pas - nas) / (pas + nas + 1e-8)

    eai = (dict_of_metrics["jawOpen"] + dict_of_metrics["mouthStretchLeft"] 
    + dict_of_metrics["mouthStretchRight"] + dict_of_metrics["mouthShrugLower"] 
    + dict_of_metrics["mouthShrugUpper"])

    ris = si + eai

    fan = np.linalg.norm(list(dict_of_metrics.values()))
    
    return [pas, nas, si, epi, eai, ris, fan]


def get_blendshapes(model_path: str , cap, target_fps: int = 5) -> pd.DataFrame:
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(math.ceil(fps / target_fps)))

    result = []

    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_id = 0
        for frame in range(n_frames):
            ret, img = cap.read()
            if ret == False:
                break

            if frame_id % step == 0:
                img_for_pipe = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_for_pipe)
                frame_timestamp_ms = int(frame_id / fps * 1000)
                face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                bs = face_landmarker_result.face_blendshapes
                if bs:
                    person = bs[0]
                    dct = {el.category_name: el.score for el in person}
                    result.append(count_l2_metrics(dct))
            frame_id += 1

    cap.release()

    if not result:
        raise ValueError("Не удалось получить ни одного кадра с лицом")
    
    return pd.DataFrame(result, columns=["PAS", "NAS", "SI", "EPI", "EAI", "RIS", "FAN"])


def calculate_l3_metrics(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in df.columns if column != "FAN"]
    rows = []

    weights = df["FAN"] / (df["FAN"].mean() + 1e-8)
    for column in columns:
        feature = df[column]

        weighted_mean = (feature * weights).sum() / (weights.sum() + 1e-8)
        std = feature.std()
        q_95 = np.quantile(feature, 0.95)
        
        thr = feature.mean() + 2 * std
        ratio_above_thr = (feature > thr).mean()
        
        rows.append([weighted_mean, std, q_95, ratio_above_thr])

    return pd.DataFrame(data=rows, columns=["weighted_mean", "std", "quantile_95", "ratio_above_thr"], index=columns)


def calculate_l4_metrics(df) -> tuple:
    agi = (
    0.6 * df.loc["SI",  "weighted_mean"] +
    0.4 * df.loc["RIS", "quantile_95"]
    )

    egi = (
    df.loc["EAI", "weighted_mean"] +
    df.loc["EAI", "std"] +
    df.loc["RIS", "ratio_above_thr"]
    )


    vsi = (
    df.loc["PAS", "weighted_mean"] -
    df.loc["NAS", "weighted_mean"]
    )

    mgi = (
    df.loc["SI",  "quantile_95"] +
    df.loc["RIS", "quantile_95"]
    )

    return agi, egi, vsi, mgi


class AdReactionAnalyzer:
    def __init__(self, model_path, baseline_results="baseline_metrics.csv"):
        self._model_path = model_path
        self._baseline_results = pd.read_csv(baseline_results, index_col='Unnamed: 0')
        self._fitted = None

    def fit(self, video_path, target_fps=5):
        cap = parse_video(video_path)
        self._target_fps = target_fps
        
        l2_df = get_blendshapes(model_path=self._model_path, cap=cap, target_fps=self._target_fps)
        l3_df = calculate_l3_metrics(l2_df)
        self._fitted = l3_df
        return self
    

    def predict(self):
        if self._fitted is None:
            raise RuntimeError("Сначала необходимо сделать fit")
        delta = self._fitted - self._baseline_results
        return calculate_l4_metrics(delta)
        