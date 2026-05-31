"""
=============================================================================
  AML Fraud Detection — Inference Benchmark & Performance Measurement
=============================================================================
Mengukur metrik kinerja komputasi model LightGBM (AML Detection):
  - CPU & Memory Usage   : peak memory (MB) + rata-rata CPU (%)
  - Inference Latency    : waktu rata-rata per inferensi (ms)
  - Memory Consumption   : footprint memori model saat runtime (MB)
  - Model Size on Disk   : ukuran file .pkl (MB)
  - Transaction Throughput: prediksi per detik (TPS)

Pustaka utama: psutil, time, threading, joblib, numpy
=============================================================================
"""

import os
import time
import json
import threading
import warnings
import numpy as np
import psutil
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "aml_best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "aml_scaler.pkl")
OUTPUT_JSON = os.path.join(BASE_DIR, "inference_metrics_updated.json")

# Nama fitur sesuai urutan training
FEATURE_NAMES = [
    "Sender_account", "Receiver_account", "Amount",
    "Payment_currency", "Received_currency",
    "Sender_bank_location", "Receiver_bank_location",
    "Payment_type", "Time_hour", "Time_dayofweek",
    "Time_month", "Time_is_weekend", "Date_hour",
    "Date_dayofweek", "Date_month", "Date_is_weekend",
    "currency_mismatch", "sender_tx_count",
    "sender_avg_amount", "sender_std_amount",
    "sender_max_amount", "amount_vs_sender_avg",
]
N_FEATURES = len(FEATURE_NAMES)

# Jumlah sampel sintetis untuk benchmark
N_SAMPLES_WARMUP     = 50    # warm-up: mengisi cache JIT LightGBM
N_SAMPLES_SINGLE     = 500   # latency single-row
N_SAMPLES_BATCH      = 5_000 # throughput batch
N_SAMPLES_STRESS     = 10_000# stress test

PRINT_WIDTH = 60

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Cetak header / divider
# ─────────────────────────────────────────────────────────────────────────────
def hdr(title: str):
    pad = (PRINT_WIDTH - len(title) - 2) // 2
    print("\n" + "═" * PRINT_WIDTH)
    print(" " * pad + f" {title} " + " " * pad)
    print("═" * PRINT_WIDTH)

def sub(label: str, value):
    print(f"  {label:<38} {value}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. UKURAN FILE MODEL DI DISK
# ─────────────────────────────────────────────────────────────────────────────
def get_model_file_size(path: str) -> float:
    """Kembalikan ukuran file dalam MB."""
    return os.path.getsize(path) / (1024 ** 2)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MEMORY FOOTPRINT MODEL SAAT RUNTIME
# ─────────────────────────────────────────────────────────────────────────────
def measure_model_memory_footprint(model_path: str, scaler_path: str) -> dict:
    """
    Mengukur selisih RSS (Resident Set Size) proses sebelum dan sesudah
    memuat model + scaler ke memori.  Pendekatan ini akurat karena
    menggunakan psutil.Process.memory_info() yang membaca langsung dari
    /proc/<pid>/status tanpa overhead profiler eksternal.
    """
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 ** 2)   # MB

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    mem_after = proc.memory_info().rss / (1024 ** 2)    # MB
    footprint = mem_after - mem_before

    return {
        "model" : model,
        "scaler": scaler,
        "rss_before_mb" : round(mem_before,  2),
        "rss_after_mb"  : round(mem_after,   2),
        "footprint_mb"  : round(max(footprint, 0), 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. MONITOR CPU & MEMORY SECARA PARALEL (background thread)
# ─────────────────────────────────────────────────────────────────────────────
class ResourceMonitor:
    """
    Polling psutil di thread terpisah setiap `interval` detik.
    Thread daemon → otomatis berhenti jika main thread selesai.

    Kenapa polling, bukan profiler blok?
      - psutil.cpu_percent(interval=None) memberikan nilai non-blocking
        berdasarkan delta dari panggilan sebelumnya, cocok untuk loop cepat.
      - Lebih ringan daripada torch.profiler atau cProfile untuk kasus
        scikit-learn / LightGBM non-GPU.
    """

    def __init__(self, interval: float = 0.05):
        self.interval   = interval
        self.cpu_samples  : list[float] = []
        self.mem_samples  : list[float] = []
        self._running   = False
        self._thread    = None
        self._proc      = psutil.Process(os.getpid())
        # inisialisasi delta CPU agar sample pertama tidak nol palsu
        self._proc.cpu_percent(interval=None)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _poll(self):
        while self._running:
            self.cpu_samples.append(self._proc.cpu_percent(interval=None))
            self.mem_samples.append(self._proc.memory_info().rss / (1024 ** 2))
            time.sleep(self.interval)

    @property
    def avg_cpu(self) -> float:
        valid = [x for x in self.cpu_samples if x > 0]
        return round(sum(valid) / len(valid), 2) if valid else 0.0

    @property
    def peak_memory_mb(self) -> float:
        return round(max(self.mem_samples), 2) if self.mem_samples else 0.0

    @property
    def avg_memory_mb(self) -> float:
        return round(sum(self.mem_samples) / len(self.mem_samples), 2) \
               if self.mem_samples else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERATE DATA SINTETIS
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(n: int) -> np.ndarray:
    """
    Membuat data acak dengan distribusi mirip data AML nyata:
    - Kolom kategorik (encoded int): Sender/Receiver account, currency, dll.
    - Kolom numerik: Amount, rata-rata, std, dsb.
    """
    rng = np.random.default_rng(seed=42)

    data = np.column_stack([
        rng.integers(1000, 9999,   size=n),          # Sender_account
        rng.integers(1000, 9999,   size=n),          # Receiver_account
        rng.exponential(5000,      size=n),          # Amount
        rng.integers(0, 10,        size=n),          # Payment_currency
        rng.integers(0, 10,        size=n),          # Received_currency
        rng.integers(0, 50,        size=n),          # Sender_bank_location
        rng.integers(0, 50,        size=n),          # Receiver_bank_location
        rng.integers(0, 5,         size=n),          # Payment_type
        rng.integers(0, 24,        size=n),          # Time_hour
        rng.integers(0, 7,         size=n),          # Time_dayofweek
        rng.integers(1, 13,        size=n),          # Time_month
        rng.integers(0, 2,         size=n),          # Time_is_weekend
        rng.integers(0, 24,        size=n),          # Date_hour
        rng.integers(0, 7,         size=n),          # Date_dayofweek
        rng.integers(1, 13,        size=n),          # Date_month
        rng.integers(0, 2,         size=n),          # Date_is_weekend
        rng.integers(0, 2,         size=n),          # currency_mismatch
        rng.integers(1, 200,       size=n),          # sender_tx_count
        rng.exponential(4000,      size=n),          # sender_avg_amount
        rng.exponential(1500,      size=n),          # sender_std_amount
        rng.exponential(20000,     size=n),          # sender_max_amount
        rng.normal(1.0, 0.5,       size=n),          # amount_vs_sender_avg
    ]).astype(np.float64)

    return data

# ─────────────────────────────────────────────────────────────────────────────
# 5. LATENCY SINGLE-ROW
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_single_latency(model, scaler, n: int) -> dict:
    """
    Mengukur waktu inferensi satu sampel pada setiap iterasi.
    Menggunakan time.perf_counter() — resolusi nanosecond,
    lebih presisi daripada time.time() untuk interval pendek.
    """
    data = generate_synthetic_data(n)
    latencies_ms = []

    for i in range(n):
        row = data[i].reshape(1, -1)
        t0 = time.perf_counter()
        scaled = scaler.transform(row)
        _      = model.predict(scaled)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1_000)

    arr = np.array(latencies_ms)
    return {
        "n_samples"       : n,
        "mean_ms"         : round(float(arr.mean()),    4),
        "median_ms"       : round(float(np.median(arr)), 4),
        "p95_ms"          : round(float(np.percentile(arr, 95)), 4),
        "p99_ms"          : round(float(np.percentile(arr, 99)), 4),
        "min_ms"          : round(float(arr.min()),     4),
        "max_ms"          : round(float(arr.max()),     4),
        "std_ms"          : round(float(arr.std()),     4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. THROUGHPUT BATCH
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_throughput(model, scaler, n: int) -> dict:
    """
    Menghitung TPS (Transactions Per Second) dengan mengukur waktu total
    untuk memprediksi N sampel sekaligus (batch inference).

    Batch inference jauh lebih efisien karena:
      - Menghindari overhead per-call Python
      - LightGBM memanfaatkan SIMD/OpenMP pada level C++
    """
    data   = generate_synthetic_data(n)
    scaled = scaler.transform(data)

    t0 = time.perf_counter()
    preds = model.predict(scaled)
    t1 = time.perf_counter()

    elapsed_s = t1 - t0
    tps = n / elapsed_s

    fraud_count  = int(preds.sum())
    normal_count = n - fraud_count

    return {
        "n_samples"      : n,
        "elapsed_s"      : round(elapsed_s, 6),
        "tps"            : round(tps, 2),
        "fraud_detected" : fraud_count,
        "normal_detected": normal_count,
        "fraud_rate_pct" : round(fraud_count / n * 100, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 7. STRESS TEST + RESOURCE MONITOR
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_stress_with_monitor(model, scaler, n: int) -> dict:
    """
    Menjalankan inferensi besar sambil memantau CPU & Memory secara real-time.
    ResourceMonitor berjalan di background thread selama inferensi berlangsung.
    """
    data   = generate_synthetic_data(n)
    scaled = scaler.transform(data)

    monitor = ResourceMonitor(interval=0.05)
    monitor.start()

    t0 = time.perf_counter()
    _  = model.predict(scaled)
    t1 = time.perf_counter()

    monitor.stop()

    elapsed_s = t1 - t0
    return {
        "n_samples"        : n,
        "elapsed_s"        : round(elapsed_s, 6),
        "tps"              : round(n / elapsed_s, 2),
        "avg_cpu_pct"      : monitor.avg_cpu,
        "peak_memory_mb"   : monitor.peak_memory_mb,
        "avg_memory_mb"    : monitor.avg_memory_mb,
        "n_cpu_samples"    : len(monitor.cpu_samples),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 8. SYSTEM INFO
# ─────────────────────────────────────────────────────────────────────────────
def get_system_info() -> dict:
    vm = psutil.virtual_memory()
    return {
        "cpu_physical_cores"  : psutil.cpu_count(logical=False),
        "cpu_logical_cores"   : psutil.cpu_count(logical=True),
        "ram_total_gb"        : round(vm.total / (1024**3), 2),
        "ram_available_gb"    : round(vm.available / (1024**3), 2),
        "python_version"      : __import__("sys").version.split()[0],
        "lightgbm_version"    : __import__("lightgbm").__version__,
        "psutil_version"      : psutil.__version__,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    hdr("AML Inference Benchmark")

    # ── 0. System info ────────────────────────────────────────────────────────
    sysinfo = get_system_info()
    hdr("System Information")
    for k, v in sysinfo.items():
        sub(k, v)

    # ── 1. Ukuran file model di disk ──────────────────────────────────────────
    model_size_mb  = get_model_file_size(MODEL_PATH)
    scaler_size_mb = get_model_file_size(SCALER_PATH)
    hdr("Model File Size on Disk")
    sub("aml_best_model.pkl", f"{model_size_mb:.3f} MB")
    sub("aml_scaler.pkl",     f"{scaler_size_mb:.3f} MB")
    sub("Total",              f"{model_size_mb + scaler_size_mb:.3f} MB")

    # ── 2. Load model & ukur memory footprint ────────────────────────────────
    hdr("Loading Model — Memory Footprint")
    print("  Memuat model + scaler ke RAM…")
    mem_info = measure_model_memory_footprint(MODEL_PATH, SCALER_PATH)
    model  = mem_info["model"]
    scaler = mem_info["scaler"]
    sub("RSS sebelum load (MB)",  mem_info["rss_before_mb"])
    sub("RSS sesudah load (MB)",  mem_info["rss_after_mb"])
    sub("Model memory footprint", f"{mem_info['footprint_mb']:.2f} MB")

    # ── 3. Warm-up ────────────────────────────────────────────────────────────
    hdr("Warm-Up Inference")
    print(f"  Menjalankan {N_SAMPLES_WARMUP} inferensi untuk mengisi cache JIT…")
    warmup_data = generate_synthetic_data(N_SAMPLES_WARMUP)
    for i in range(N_SAMPLES_WARMUP):
        model.predict(scaler.transform(warmup_data[i].reshape(1, -1)))
    print("  Warm-up selesai ✓")

    # ── 4. Latency single-row ─────────────────────────────────────────────────
    hdr("Inference Latency (Single-Row)")
    print(f"  Mengukur {N_SAMPLES_SINGLE} inferensi satu per satu…")
    lat = benchmark_single_latency(model, scaler, N_SAMPLES_SINGLE)
    sub("Jumlah sampel",   lat["n_samples"])
    sub("Mean latency",    f"{lat['mean_ms']:.4f} ms")
    sub("Median latency",  f"{lat['median_ms']:.4f} ms")
    sub("P95 latency",     f"{lat['p95_ms']:.4f} ms")
    sub("P99 latency",     f"{lat['p99_ms']:.4f} ms")
    sub("Min latency",     f"{lat['min_ms']:.4f} ms")
    sub("Max latency",     f"{lat['max_ms']:.4f} ms")
    sub("Std dev",         f"{lat['std_ms']:.4f} ms")

    # ── 5. Throughput batch ───────────────────────────────────────────────────
    hdr("Transaction Throughput (Batch)")
    print(f"  Memprediksi {N_SAMPLES_BATCH:,} sampel sekaligus…")
    tput = benchmark_throughput(model, scaler, N_SAMPLES_BATCH)
    sub("Jumlah sampel",    f"{tput['n_samples']:,}")
    sub("Elapsed time",     f"{tput['elapsed_s']:.4f} s")
    sub("Throughput (TPS)", f"{tput['tps']:,.1f} transaksi/detik")
    sub("Fraud terdeteksi", f"{tput['fraud_detected']:,}  ({tput['fraud_rate_pct']}%)")
    sub("Normal terdeteksi",f"{tput['normal_detected']:,}")

    # ── 6. Stress test + CPU/Mem monitor ─────────────────────────────────────
    hdr("Stress Test + CPU & Memory Monitor")
    print(f"  Inferensi {N_SAMPLES_STRESS:,} sampel + polling resource…")
    stress = benchmark_stress_with_monitor(model, scaler, N_SAMPLES_STRESS)
    sub("Jumlah sampel",        f"{stress['n_samples']:,}")
    sub("Elapsed time",         f"{stress['elapsed_s']:.4f} s")
    sub("Throughput (TPS)",     f"{stress['tps']:,.1f} transaksi/detik")
    sub("Avg CPU usage",        f"{stress['avg_cpu_pct']:.2f} %")
    sub("Peak memory usage",    f"{stress['peak_memory_mb']:.2f} MB")
    sub("Avg memory usage",     f"{stress['avg_memory_mb']:.2f} MB")
    sub("Monitor samples (n)",  stress["n_cpu_samples"])

    # ── 7. Simpan hasil ke JSON ───────────────────────────────────────────────
    results = {
        "project"         : "AML Fraud Detection — LightGBM Inference Benchmark",
        "system_info"     : sysinfo,
        "model_on_disk"   : {
            "model_pkl_mb" : round(model_size_mb,  3),
            "scaler_pkl_mb": round(scaler_size_mb, 3),
            "total_mb"     : round(model_size_mb + scaler_size_mb, 3),
        },
        "memory_footprint_mb" : mem_info["footprint_mb"],
        "latency"         : lat,
        "throughput_batch": tput,
        "stress_test"     : stress,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    hdr("Hasil Tersimpan")
    print(f"  → {OUTPUT_JSON}\n")

    # ── 8. Panduan interpretasi ───────────────────────────────────────────────
    hdr("Panduan Interpretasi Metrik")
    guide = """
  ┌──────────────────────────────────────────────────────────┐
  │  METRIK          │ NILAI BAIK         │ INTERPRETASI      │
  ├──────────────────┼────────────────────┼───────────────────┤
  │ Inference Latency│ < 5 ms (real-time) │ Lebih rendah →    │
  │  (mean / P99)    │ < 50 ms (batch ok) │ respons lebih      │
  │                  │                    │ cepat, cocok       │
  │                  │                    │ untuk streaming    │
  ├──────────────────┼────────────────────┼───────────────────┤
  │ Memory Footprint │ < 500 MB           │ Lebih kecil →      │
  │                  │                    │ bisa deploy di     │
  │                  │                    │ edge / container   │
  ├──────────────────┼────────────────────┼───────────────────┤
  │ Model Size (disk)│ < 50 MB            │ Penting untuk      │
  │                  │                    │ CI/CD & transfer   │
  ├──────────────────┼────────────────────┼───────────────────┤
  │ TPS (Throughput) │ > 1.000 TPS        │ Lebih tinggi →     │
  │                  │                    │ mampu menangani    │
  │                  │                    │ volume transaksi   │
  │                  │                    │ bank besar         │
  ├──────────────────┼────────────────────┼───────────────────┤
  │ CPU Usage (avg)  │ < 40 % per core    │ Headroom tersisa   │
  │                  │                    │ untuk beban lain   │
  ├──────────────────┼────────────────────┼───────────────────┤
  │ Peak Memory      │ < 2× avg memory    │ Spike kecil →      │
  │                  │                    │ alokasi stabil,    │
  │                  │                    │ aman di produksi   │
  └──────────────────┴────────────────────┴───────────────────┘

  CATATAN:
  • P95/P99 latency lebih penting dari mean untuk SLA production.
  • Batch inference selalu jauh lebih efisien (SIMD, OpenMP LightGBM).
  • Warm-up diperlukan agar JIT cache terisi; tanpanya latency
    pertama bisa 5–10× lebih lambat (cold-start overhead).
  • Jika TPS < kebutuhan bisnis, pertimbangkan:
      1. Prediksi batch lebih besar (vectorized)
      2. Kurangi fitur (feature selection)
      3. Model distillation / quantization
      4. Deploy dengan ONNX Runtime atau Treelite
    """
    print(guide)

    return results


if __name__ == "__main__":
    main()
