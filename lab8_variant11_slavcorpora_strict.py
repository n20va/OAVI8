from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

matplotlib.use("Agg")
plt.rcParams["font.family"] = "DejaVu Sans"

ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

VARIANT = 11
METHOD_NAME = "LBP (Local Binary Pattern)"
FEATURE_NAME = "H(LBP)"
BRIGHTNESS_TRANSFORM = "Выравнивание гистограммы яркости"

# Изображения берутся из выборки "Жесть".
# При необходимости можно заменить индексы на другие из этой же выборки.
IMAGE_INDICES = [0, 5, 10]

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_variant11"
SRC_DIR = BASE_DIR / "src_variant11"
REPORT_PATH = BASE_DIR / "report-8-variant11.md"

LBP_POINTS = 8
LBP_LEVELS = 256

# Уменьшение веса изображений для GitHub.
MAX_IMAGE_SIDE = 800
JPEG_QUALITY = 45
PLOT_DPI = 95


@dataclass
class CaseResult:
    case_no: int
    image_index: int
    source_url: str
    width: int
    height: int
    source_name: str
    gray_name: str
    contrast_gray_name: str
    contrast_color_name: str
    hist_name: str
    lbp_before_name: str
    lbp_after_name: str
    lbp_hist_name: str
    csv_name: str
    entropy_before: float
    entropy_after: float
    euclidean_distance: float
    l1_distance: float
    cosine_similarity: float


def fetch_image_paths(origin: str, sample_id: str) -> list[str]:
    response = requests.get(f"{origin}/api/samples/{sample_id}", timeout=30)
    response.raise_for_status()
    sample_data = response.json()
    return [f"{origin}/images/{page['filename']}" for page in sample_data["pages"]]


def download_image_rgb(image_url: str) -> np.ndarray:
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.asarray(pil_image, dtype=np.uint8)


def _downscale_for_report(pil_image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Уменьшает изображение для отчёта, чтобы файлы нормально загружались на GitHub."""
    w, h = pil_image.size
    if max(w, h) <= max_side:
        return pil_image
    scale = max_side / float(max(w, h))
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return pil_image.resize(new_size, Image.Resampling.LANCZOS)


def save_rgb(image: np.ndarray, path: Path) -> None:
    pil_image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
    pil_image = _downscale_for_report(pil_image).convert("RGB")
    pil_image.save(path, quality=JPEG_QUALITY, optimize=True, progressive=True)


def save_gray(image: np.ndarray, path: Path) -> None:
    pil_image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L")
    pil_image = _downscale_for_report(pil_image).convert("L")
    pil_image.save(path, quality=JPEG_QUALITY, optimize=True, progressive=True)


def rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgbf = rgb.astype(np.float32) / 255.0
    r = rgbf[..., 0]
    g = rgbf[..., 1]
    b = rgbf[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    l = 0.5 * (maxc + minc)

    s = np.zeros_like(l, dtype=np.float32)
    denom = 1.0 - np.abs(2.0 * l - 1.0)
    mask = delta > 1e-8
    s[mask] = delta[mask] / np.maximum(denom[mask], 1e-8)

    h = np.zeros_like(l, dtype=np.float32)
    mask_r = mask & (maxc == r)
    mask_g = mask & (maxc == g)
    mask_b = mask & (maxc == b)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / np.maximum(delta[mask_r], 1e-8)) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / np.maximum(delta[mask_g], 1e-8)) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / np.maximum(delta[mask_b], 1e-8)) + 4.0
    h = (h / 6.0) % 1.0

    return h, s, l


def hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hh = (h % 1.0) * 6.0
    x = c * (1.0 - np.abs((hh % 2.0) - 1.0))
    m = l - 0.5 * c

    r1 = np.zeros_like(h, dtype=np.float32)
    g1 = np.zeros_like(h, dtype=np.float32)
    b1 = np.zeros_like(h, dtype=np.float32)

    c0 = (hh >= 0.0) & (hh < 1.0)
    c1 = (hh >= 1.0) & (hh < 2.0)
    c2 = (hh >= 2.0) & (hh < 3.0)
    c3 = (hh >= 3.0) & (hh < 4.0)
    c4 = (hh >= 4.0) & (hh < 5.0)
    c5 = (hh >= 5.0) & (hh <= 6.0)

    r1[c0], g1[c0], b1[c0] = c[c0], x[c0], 0.0
    r1[c1], g1[c1], b1[c1] = x[c1], c[c1], 0.0
    r1[c2], g1[c2], b1[c2] = 0.0, c[c2], x[c2]
    r1[c3], g1[c3], b1[c3] = 0.0, x[c3], c[c3]
    r1[c4], g1[c4], b1[c4] = x[c4], 0.0, c[c4]
    r1[c5], g1[c5], b1[c5] = c[c5], 0.0, x[c5]

    rgb = np.stack([r1 + m, g1 + m, b1 + m], axis=-1)
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def histogram_equalization_uint8(gray: np.ndarray) -> np.ndarray:
    hist = np.bincount(gray.ravel(), minlength=LBP_LEVELS).astype(np.float64)
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return gray.copy()

    cdf_min = nonzero[0]
    total = gray.size
    if total == cdf_min:
        return gray.copy()

    lut = np.rint((cdf - cdf_min) / (total - cdf_min) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[gray]


def equalize_l_channel(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, s, l = rgb_to_hsl(rgb)
    gray_before = np.rint(np.clip(l * 255.0, 0.0, 255.0)).astype(np.uint8)
    gray_after = histogram_equalization_uint8(gray_before)
    l_after = gray_after.astype(np.float32) / 255.0
    rgb_after = hsl_to_rgb(h, s, l_after)
    return gray_before, gray_after, rgb_after


def compute_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Классический LBP 3x3:
    соседний пиксель получает 1, если он >= центрального пикселя.
    Код формируется из 8 соседей по часовой стрелке, начиная с левого верхнего.
    """
    g = gray.astype(np.uint8)
    center = g[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)

    neighbors = [
        g[:-2, :-2],   # верх-лево
        g[:-2, 1:-1],  # верх
        g[:-2, 2:],    # верх-право
        g[1:-1, 2:],   # право
        g[2:, 2:],     # низ-право
        g[2:, 1:-1],   # низ
        g[2:, :-2],    # низ-лево
        g[1:-1, :-2],  # лево
    ]

    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    return lbp


def lbp_histogram(lbp: np.ndarray) -> np.ndarray:
    hist = np.bincount(lbp.ravel(), minlength=LBP_LEVELS).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return hist
    return hist / total


def entropy(hist: np.ndarray) -> float:
    p = hist[hist > 0]
    return float(-np.sum(p * np.log2(p)))


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.rint((arr - vmin) * 255.0 / (vmax - vmin)).clip(0, 255).astype(np.uint8)


def plot_brightness_histograms(gray_before: np.ndarray, gray_after: np.ndarray, out_path: Path) -> None:
    hist_before, _ = np.histogram(gray_before, bins=256, range=(0, 256))
    hist_after, _ = np.histogram(gray_after, bins=256, range=(0, 256))

    plt.figure(figsize=(10, 4))
    plt.plot(hist_before, label="До выравнивания", linewidth=1.2)
    plt.plot(hist_after, label="После выравнивания", linewidth=1.2)
    plt.title("Гистограммы яркости")
    plt.xlabel("Уровень яркости")
    plt.ylabel("Количество пикселей")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


def plot_lbp_histograms(hist_before: np.ndarray, hist_after: np.ndarray, out_path: Path) -> None:
    x = np.arange(LBP_LEVELS)
    plt.figure(figsize=(11, 4.5))
    plt.bar(x - 0.2, hist_before, width=0.4, label="До")
    plt.bar(x + 0.2, hist_after, width=0.4, label="После")
    plt.title("H(LBP): нормированные гистограммы LBP")
    plt.xlabel("LBP-код")
    plt.ylabel("Доля")
    plt.xlim(-1, 256)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


def save_lbp_hist_csv(path: Path, hist_before: np.ndarray, hist_after: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["lbp_code", "h_lbp_before", "h_lbp_after", "delta"])
        for i, (before, after) in enumerate(zip(hist_before, hist_after)):
            writer.writerow([i, f"{before:.10f}", f"{after:.10f}", f"{after - before:.10f}"])


def cleanup_generated_files(directory: Path) -> None:
    if not directory.exists():
        return
    for file_path in directory.glob("img*"):
        if file_path.is_file():
            file_path.unlink()
    for name in ["summary.csv", "summary.json"]:
        p = directory / name
        if p.exists():
            p.unlink()


def process_case(case_no: int, image_index: int, source_url: str) -> CaseResult:
    source_rgb = download_image_rgb(source_url)
    height, width = source_rgb.shape[:2]

    gray_before, gray_after, contrast_rgb = equalize_l_channel(source_rgb)

    lbp_before = compute_lbp(gray_before)
    lbp_after = compute_lbp(gray_after)
    hist_lbp_before = lbp_histogram(lbp_before)
    hist_lbp_after = lbp_histogram(lbp_after)

    entropy_before = entropy(hist_lbp_before)
    entropy_after = entropy(hist_lbp_after)
    euclidean_distance = float(np.linalg.norm(hist_lbp_after - hist_lbp_before))
    l1_distance = float(np.sum(np.abs(hist_lbp_after - hist_lbp_before)))
    cosine_similarity = float(
        np.dot(hist_lbp_before, hist_lbp_after)
        / (np.linalg.norm(hist_lbp_before) * np.linalg.norm(hist_lbp_after) + 1e-12)
    )

    source_name = f"img{case_no}_source.jpg"
    gray_name = f"img{case_no}_gray_before.jpg"
    contrast_gray_name = f"img{case_no}_gray_equalized.jpg"
    contrast_color_name = f"img{case_no}_color_equalized.jpg"
    hist_name = f"img{case_no}_brightness_histograms.jpg"
    lbp_before_name = f"img{case_no}_lbp_before.jpg"
    lbp_after_name = f"img{case_no}_lbp_after.jpg"
    lbp_hist_name = f"img{case_no}_lbp_histograms.jpg"
    csv_name = f"img{case_no}_lbp_histograms.csv"

    for directory in (RESULTS_DIR, SRC_DIR):
        save_rgb(source_rgb, directory / source_name)
        save_gray(gray_before, directory / gray_name)
        save_gray(gray_after, directory / contrast_gray_name)
        save_rgb(contrast_rgb, directory / contrast_color_name)
        save_gray(lbp_before, directory / lbp_before_name)
        save_gray(lbp_after, directory / lbp_after_name)
        plot_brightness_histograms(gray_before, gray_after, directory / hist_name)
        plot_lbp_histograms(hist_lbp_before, hist_lbp_after, directory / lbp_hist_name)
        save_lbp_hist_csv(directory / csv_name, hist_lbp_before, hist_lbp_after)

    return CaseResult(
        case_no=case_no,
        image_index=image_index,
        source_url=source_url,
        width=width,
        height=height,
        source_name=source_name,
        gray_name=gray_name,
        contrast_gray_name=contrast_gray_name,
        contrast_color_name=contrast_color_name,
        hist_name=hist_name,
        lbp_before_name=lbp_before_name,
        lbp_after_name=lbp_after_name,
        lbp_hist_name=lbp_hist_name,
        csv_name=csv_name,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        euclidean_distance=euclidean_distance,
        l1_distance=l1_distance,
        cosine_similarity=cosine_similarity,
    )


def write_summary(cases: list[CaseResult]) -> None:
    rows = []
    for case in cases:
        rows.append(
            {
                "case_no": case.case_no,
                "image_index": case.image_index,
                "source_url": case.source_url,
                "size": f"{case.width}x{case.height}",
                "entropy_before": round(case.entropy_before, 8),
                "entropy_after": round(case.entropy_after, 8),
                "euclidean_distance": round(case.euclidean_distance, 8),
                "l1_distance": round(case.l1_distance, 8),
                "cosine_similarity": round(case.cosine_similarity, 8),
            }
        )

    for directory in (RESULTS_DIR, SRC_DIR):
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "variant": VARIANT,
                    "method": METHOD_NAME,
                    "feature": FEATURE_NAME,
                    "brightness_transform": BRIGHTNESS_TRANSFORM,
                    "cases": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                [
                    "case_no",
                    "image_index",
                    "source_url",
                    "size",
                    "entropy_before",
                    "entropy_after",
                    "euclidean_distance",
                    "l1_distance",
                    "cosine_similarity",
                ]
            )
            for case in cases:
                writer.writerow(
                    [
                        case.case_no,
                        case.image_index,
                        case.source_url,
                        f"{case.width}x{case.height}",
                        f"{case.entropy_before:.10f}",
                        f"{case.entropy_after:.10f}",
                        f"{case.euclidean_distance:.10f}",
                        f"{case.l1_distance:.10f}",
                        f"{case.cosine_similarity:.10f}",
                    ]
                )


def write_report(cases: list[CaseResult]) -> None:
    lines: list[str] = []
    lines.append("# Лабораторная работа №8")
    lines.append("## Текстурный анализ и контрастирование")
    lines.append("")
    lines.append("### Исходные данные")
    lines.append("- Вариант: 11")
    lines.append("- Матрица/метод: LBP")
    lines.append("- Расчёт признаков: H(LBP)")
    lines.append("- Преобразование яркости: выравнивание гистограммы")
    lines.append('- Выборка: "Жесть", slavcorpora.ru')
    lines.append("")
    lines.append("| № | Индекс в выборке | Размер | Источник |")
    lines.append("|:-:|:----------------:|-------:|:---------|")
    for case in cases:
        lines.append(f"| {case.case_no} | {case.image_index} | {case.width}x{case.height} | `{case.source_url}` |")
    lines.append("")
    lines.append("### Результаты обработки")
    lines.append("")

    for case in cases:
        lines.append(f"#### Изображение {case.case_no}")
        lines.append("")
        lines.append("| Исходное | Полутоновое | После выравнивания |")
        lines.append("|:--------:|:-----------:|:-------------------:|")
        lines.append(f"| ![source](src_variant11/{case.source_name}) | ![gray](src_variant11/{case.gray_name}) | ![eq](src_variant11/{case.contrast_gray_name}) |")
        lines.append("")
        lines.append("| Гистограммы яркости | LBP до | LBP после |")
        lines.append("|:-------------------:|:------:|:---------:|")
        lines.append(f"| ![hist](src_variant11/{case.hist_name}) | ![lbp_before](src_variant11/{case.lbp_before_name}) | ![lbp_after](src_variant11/{case.lbp_after_name}) |")
        lines.append("")
        lines.append(f"![lbp_hist](src_variant11/{case.lbp_hist_name})")
        lines.append("")
        lines.append("| Показатель | Значение |")
        lines.append("|:-----------|---------:|")
        lines.append(f"| Энтропия H(LBP) до | `{case.entropy_before:.6f}` |")
        lines.append(f"| Энтропия H(LBP) после | `{case.entropy_after:.6f}` |")
        lines.append(f"| Евклидово расстояние | `{case.euclidean_distance:.6f}` |")
        lines.append(f"| L1-расстояние | `{case.l1_distance:.6f}` |")
        lines.append(f"| Косинусное сходство | `{case.cosine_similarity:.6f}` |")
        lines.append(f"| CSV | `src_variant11/{case.csv_name}` |")
        lines.append("")

    lines.append("### Сводная таблица")
    lines.append("")
    lines.append("| № | Размер | Entropy до | Entropy после | Euclid | L1 | Cos |")
    lines.append("|:-:|:------:|-----------:|--------------:|-------:|---:|----:|")
    for case in cases:
        lines.append(
            f"| {case.case_no} | {case.width}x{case.height} | {case.entropy_before:.6f} | {case.entropy_after:.6f} | {case.euclidean_distance:.6f} | {case.l1_distance:.6f} | {case.cosine_similarity:.6f} |"
        )
    lines.append("")
    lines.append("Сводные файлы: `src_variant11/summary.csv`, `src_variant11/summary.json`.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    print("Создаём папки...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_generated_files(RESULTS_DIR)
    cleanup_generated_files(SRC_DIR)

    print("Получаем список изображений из выборки...")
    image_paths = fetch_image_paths(ORIGIN, SAMPLE_ID)
    if not image_paths:
        raise RuntimeError("Список изображений пуст.")
    print(f"Найдено изображений: {len(image_paths)}")

    cases: list[CaseResult] = []
    for case_no, image_index in enumerate(IMAGE_INDICES, start=1):
        if image_index < 0 or image_index >= len(image_paths):
            raise IndexError(f"Индекс {image_index} выходит за пределы списка image_paths.")

        print(f"\nОбрабатывается изображение {case_no} (индекс {image_index})...")
        source_url = image_paths[image_index]
        cases.append(process_case(case_no, image_index, source_url))

    print("\nФормируем отчёт...")
    write_summary(cases)
    write_report(cases)

    print("\nЛабораторная работа №8 выполнена.")
    print(f"Вариант: {VARIANT} ({METHOD_NAME})")
    print(f"Признак: {FEATURE_NAME}")
    print(f"Преобразование яркости: {BRIGHTNESS_TRANSFORM}")
    print(f"Результаты: {RESULTS_DIR}")
    print(f"Файлы для отчёта: {SRC_DIR}")
    print(f"Отчёт: {REPORT_PATH}")
    print(f"Изображения сохранены в JPEG, max side={MAX_IMAGE_SIDE}, quality={JPEG_QUALITY}.")


if __name__ == "__main__":
    main()
